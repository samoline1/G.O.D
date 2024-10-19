import os
from datetime import datetime
from typing import Dict
from typing import List
from typing import Tuple
from urllib.parse import urlparse

import aiohttp
import numpy as np
from fiber.logging_utils import get_logger

import validator.core.constants as cts
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.models import Submission
from validator.core.models import Task
from validator.db.sql import add_submission
from validator.db.sql import get_miners_assigned_to_task
from validator.db.sql import set_task_node_quality_score
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_get


logger = get_logger(__name__)

from scipy.stats import gmean


async def download_s3_file(file_url: str) -> str:
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    local_file_path = os.path.join("/tmp", file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path

# NOTE: doc strings are a bit long. Don't need to explain params and returns types - these
# should be clear from the parameter names and function name
def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """
    Calculate a weighted average of test and synthetic losses.

    This function combines the test loss and synthetic loss into a single metric,
    giving more weight to the test loss as defined by TEST_SCORE_WEIGHTING.
    """
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss


def calculate_scaled_score(weighted_loss: float, is_finetune: bool, scale_factor: float) -> float:
    """
    Calculate a score for a miner based on their weighted loss, using an exponential decay function.

    This function converts a loss value into a score, where lower losses result in higher scores.

    # NOTE: Im not sure what this means

    The relationship between loss and score is not linear, but exponential, meaning small
    differences in low losses can lead to larger differences in scores than the same
    difference in higher losses.

    # NOTE: don't need this - it should be clear from parameter names
    Parameters:
    weighted_loss : float
        The combined loss for the miner. Lower values indicate better performance.
    is_finetune : bool
        Indicates whether the model is finetuned. Non-finetuned models always receive a score of 0.
    scale_factor : float
        A factor that adjusts the sensitivity of the scoring to differences in loss.
        Higher scale factors amplify the effect of small loss differences on the final score.
        See compute_adaptive_scale_factor for details

    #NOTE: dont need this
    Returns:
    float
        The calculated score, ranging from 0 to 1. Higher scores indicate better performance.

    # NOTE: This should be obvious from the code
    Behavior:
    1. For non-finetuned models (is_finetune = False), the function always returns 0.
    2. As weighted_loss increases, the score decreases exponentially.
    3. The scale_factor determines how sharply the score drops off as loss increases.

    # NOTE: this sounds like AI... why are you using my mental capacity
    # to teach me about general machine learning concepts
    # when reviewing the code

    Note:
    The exponential nature of this scoring system means it's particularly good at
    distinguishing between small differences in low loss values, which is often
    desirable in machine learning contexts where small improvements in already good
    models can be significant.
    """
    base_score = np.exp(-weighted_loss * scale_factor)
    return base_score if is_finetune else 0.0


def calculate_relative_scores(task_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalise scores relative to their geometric mean.

    This function adjusts all scores so that their geometric mean becomes 1.
    This helps in comparing scores across different tasks.

    By dividing each miner's score by the geometric mean of all scores for that task,
    we're essentially measuring each miner's performance relative to the overall performance on that specific task.

    This normalisation makes the scores scale-independent.
    If Task A is inherently more difficult and results in lower scores overall,
    dividing by the geometric mean will adjust for this.
    """
    scores = np.array(list(task_scores.values()))
    geometric_mean = gmean(scores)
    relative_scores = np.where(scores > 0, scores / geometric_mean, 0)
    return {str(miner_id): float(score) for miner_id, score in zip(task_scores.keys(), relative_scores)}


def compute_adaptive_scale_factor(miner_results: List[Tuple[str, float, float, bool]]) -> float:
    """
    Compute an adaptive scale factor based on the range of losses.


    We want to calculate a scaling factor that can be applied to these scores to make them more meaningful, especially
    when the scores are closely clustered.
    For instance, if all scores fall between 0.8 and 0.85, it's difficult to distinguish performance differences.
    The function determines how much to "stretch" this range by computing a scale factor.
    This factor is calculated based on the lowest and highest scores in the set,
    aiming to create a consistent ratio between the best and worst scores (defined by a target ratio, typically 2:1).
    If the scores are tightly grouped, the scaling factor will be larger to amplify small differences.
    Conversely, if the scores are already well spread out, the scaling factor will be smaller.

    Examples:
    1. Closely clustered scores:
       miner_results = [
           ("Miner1", 0.82, 0.81, True),
           ("Miner2", 0.83, 0.82, True),
           ("Miner3", 0.81, 0.80, True),
           ("Miner4", 0.84, 0.83, True)
       ]
       Result: scale_factor ≈ 13.8
       (High scale factor due to tightly clustered scores)

    2. More spread out scores:
       miner_results = [
           ("Miner1", 0.5, 0.5, True),
           ("Miner2", 0.7, 0.7, True),
           ("Miner3", 0.9, 0.9, True),
           ("Miner4", 1.1, 1.1, True)
       ]
       Result: scale_factor ≈ 1.2
       (Lower scale factor due to already spread out scores)

    """
    weighted_losses = [calculate_weighted_loss(test_loss, synth_loss) for _, test_loss, synth_loss, _ in miner_results]
    min_loss, max_loss = min(weighted_losses), max(weighted_losses)

    if min_loss == max_loss:
        return 1.0  # Default to 1 if all losses are the same

    return np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss)


def score_adjustment(miner_results: List[Tuple[str, float, float, bool]]) -> Dict[str, float]:
    scale_factor = compute_adaptive_scale_factor(miner_results)  # see function def for details
    task_results = {}
    for miner_id, test_loss, synth_loss, is_finetune in miner_results:
        weighted_loss = calculate_weighted_loss(test_loss, synth_loss)
        task_results[miner_id] = calculate_scaled_score(weighted_loss, is_finetune, scale_factor)
    return task_results


async def evaluate_and_score(task: Task, config) -> Task:
    miner_pool = await get_miners_assigned_to_task(str(task.task_id), config.psql_db)
    task_results = []
    submission_repos = {}
    dataset_type = CustomDatasetType(
        field_system=task.system, field_instruction=task.instruction, field_input=task.input, field_output=task.output
    )

    for miner in miner_pool:
        try:
            url = f"{miner.ip}:{miner.port}/get_latest_model_submission/{task.task_id}"
            submission_repo = await process_non_stream_get(url, None)
            current_time = datetime.now()
            submission_repos[str(miner.node_id)] = Submission(
                task_id=task.task_id,
                node_id=miner.node_id,
                repo=submission_repo,
                created_on=current_time,
                updated_on=current_time,
            )
            evaluation_params = {
                "file_format": FileFormat.JSON,
                "original_model": task.model_id,
                "model": submission_repo,
                "dataset_type": dataset_type,
            }
            synthetic_data_filepath = await download_s3_file(task.synthetic_data)
            test_data_filepath = await download_s3_file(task.test_data)

            is_test_finetune, synth_loss_tuple, synth_perplexity_tuple = run_evaluation_docker(
                dataset=synthetic_data_filepath, **evaluation_params
            )
            is_synth_finetune, test_loss_tuple, test_perplexity_tuple = run_evaluation_docker(
                dataset=test_data_filepath, **evaluation_params
            )

            synth_loss = synth_loss_tuple[1]  # Assuming ('eval_loss', value)
            test_loss = test_loss_tuple[1]  # Assuming ('eval_loss', value)

            synth_perplexity = synth_perplexity_tuple[1]  # Assuming ('perplexity', value)
            test_perplexity = test_perplexity_tuple[1]  # Assuming ('perplexity', value)

            logger.info(f"The losses that we have out from {miner.node_id} are synth: {synth_loss} and test {test_loss}")
            logger.info(
                f"The perplexities that we have out from {miner.node_id} are synth: {synth_perplexity} and test {test_perplexity}"
            )

            task_results.append((miner.node_id, test_loss, synth_loss, is_test_finetune))

        except Exception as e:
            logger.info(f"There was an issue with scoring {e}")

    raw_scores = score_adjustment(task_results)
    relative_scores = calculate_relative_scores(raw_scores)
    logger.info(f"The final scores are {relative_scores} from the raw scores of {task_results}")
    logger.info(f"The sumissions are {submission_repos}")
    for miner_id, score in relative_scores.items():
        await set_task_node_quality_score(task.task_id, miner_id, score, config.psql_db)
        submission = submission_repos[miner_id]
        submission.score = score
        await add_submission(submission, config.psql_db)

    task.status = TaskStatus.SUCCESS

    return task
