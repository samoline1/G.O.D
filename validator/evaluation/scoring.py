from datetime import datetime
from scipy.stats import gmean

import numpy as np
from fiber.logging_utils import get_logger

import validator.core.constants as cts
from core.utils import download_s3_file
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Submission
from validator.core.models import Task
from validator.core.models import MinerResults
from validator.db.sql import add_submission
from validator.db.sql import get_miners_assigned_to_task
from validator.db.sql import set_task_node_quality_score
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_get


logger = get_logger(__name__)

def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """
    Calculate a weighted average of test and synthetic losses.
    """
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss

def calculate_scaled_score(weighted_loss: float, is_finetune: bool, scale_factor: float) -> float:
    """
    Calculate a score based on weighted loss using exponential decay.
    Returns 0.0 for non-finetuned models.
    """
    return np.exp(-weighted_loss * scale_factor) if is_finetune else 0.0

def adjust_miner_scores_to_be_relative_to_other_comps(miner_results: list[MinerResults]) -> list[MinerResults]:
    """
    Adjusts valid scores to have a geometric mean of 1.0.
    Submissions without valid scores (None or non-finetuned) receive a score of 0.0.
    """
    # Filter for valid results (has submission and is finetuned)
    valid_results = [
        res for res in miner_results
        if res.submission is not None
        and res.is_finetune
        and not np.isnan(res.score)
    ]

    if not valid_results:
        logger.warning("No valid submissions found. Setting all scores to 0.0")
        for res in miner_results:
            res.score = 0.0
        return miner_results

    # Calculate geometric mean of valid scores
    valid_scores = [res.score for res in valid_results]
    geometric_mean = gmean(np.array(valid_scores))

    if np.isnan(geometric_mean) or np.isinf(geometric_mean) or geometric_mean <= 0:
        logger.warning(f"Invalid geometric mean: {geometric_mean}. Setting to 1.0")
        geometric_mean = 1.0

    # Adjust scores
    for res in miner_results:
        if res.submission is None or not res.is_finetune:
            res.score = 0.0
        else:
            res.score /= geometric_mean

    return miner_results

def compute_adaptive_scale_factor(miner_results: list[MinerResults]) -> float:
    """
    Compute scale factor based on valid submissions only.
    """
    valid_results = [
        res for res in miner_results
        if res.submission is not None
        and not np.isnan(res.test_loss)
        and not np.isnan(res.synth_loss)
    ]

    if not valid_results:
        return 1.0

    weighted_losses = [calculate_weighted_loss(res.test_loss, res.synth_loss) for res in valid_results]
    min_loss, max_loss = min(weighted_losses), max(weighted_losses)

    if min_loss == max_loss:
        return 1.0

    return np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss)

def add_raw_scores_to_miner_results(miner_results: list[MinerResults]) -> list[MinerResults]:
    """
    Calculate raw scores for all submissions.
    None submissions or non-finetuned models receive a score of 0.0.
    """
    # Filter for valid results first
    valid_results = [
        res for res in miner_results
        if res.submission is not None
        and not np.isnan(res.test_loss)
        and not np.isnan(res.synth_loss)
    ]

    if not valid_results:
        logger.warning("No valid results found. Setting all scores to 0.0")
        for result in miner_results:
            result.score = 0.0
        return miner_results

    scale_factor = compute_adaptive_scale_factor(valid_results)

    for result in miner_results:
        if (result.submission is None or
            np.isnan(result.test_loss) or
            np.isnan(result.synth_loss) or
            not result.is_finetune):
            result.score = 0.0
        else:
            weighted_loss = calculate_weighted_loss(result.test_loss, result.synth_loss)
            result.score = calculate_scaled_score(weighted_loss, result.is_finetune, scale_factor)

    return miner_results

async def evaluate_and_score(task: Task, config: Config) -> Task:
    miner_pool = await get_miners_assigned_to_task(str(task.task_id), config.psql_db)
    assert task.task_id is not None
    task_results = []
    dataset_type = CustomDatasetType(
        field_system=task.system,
        field_instruction=task.instruction,
        field_input=task.input,
        field_output=task.output
    )

    for miner in miner_pool:
        try:
            url = f"{miner.ip}:{miner.port}/get_latest_model_submission/{task.task_id}"
            try:
                submission_repo = str(await process_non_stream_get(url, None))
            except Exception as e:
                logger.error(f"Failed to process non-stream get for miner {miner.node_id} - {e}")
                # Create result with zero scores and no submission
                miner_result = MinerResults(
                    node_id=miner.node_id,
                    test_loss=np.nan,
                    synth_loss=np.nan,
                    is_finetune=False,
                    submission=None
                )
                task_results.append(miner_result)
                continue

            if submission_repo is None:
                # Create result with zero scores and no submission
                miner_result = MinerResults(
                    node_id=miner.node_id,
                    test_loss=np.nan,
                    synth_loss=np.nan,
                    is_finetune=False,
                    submission=None
                )
                task_results.append(miner_result)
                continue

            # Process valid submission
            current_time = datetime.now()
            submission = Submission(
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

            assert task.synthetic_data is not None
            assert task.test_data is not None

            synthetic_data_filepath = await download_s3_file(task.synthetic_data)
            test_data_filepath = await download_s3_file(task.test_data)

            synth_eval_result = await run_evaluation_docker(
                dataset=synthetic_data_filepath, **evaluation_params
            )
            test_eval_result = await run_evaluation_docker(
                dataset=test_data_filepath, **evaluation_params
            )

            logger.info(
                f"Results for {miner.node_id} - "
                f"synth_loss: {synth_eval_result.eval_loss}, "
                f"test_loss: {test_eval_result.eval_loss}"
            )

            miner_result = MinerResults(
                node_id=miner.node_id,
                test_loss=test_eval_result.eval_loss,
                synth_loss=synth_eval_result.eval_loss,
                is_finetune=test_eval_result.is_finetune,
                submission=submission
            )
            task_results.append(miner_result)

        except Exception as e:
            logger.error(f"Error processing miner {miner.node_id}: {e}")
            # Create result with zero scores and no submission on error
            miner_result = MinerResults(
                node_id=miner.node_id,
                test_loss=np.nan,
                synth_loss=np.nan,
                is_finetune=False,
                submission=None
            )
            task_results.append(miner_result)

    # Process scores
    task_results = add_raw_scores_to_miner_results(task_results)
    task_results = adjust_miner_scores_to_be_relative_to_other_comps(task_results)

    # Update database
    for result in task_results:
        assert result.score is not None
        await set_task_node_quality_score(task.task_id, result.node_id, result.score, config.psql_db)
        if result.submission is not None:
            result.submission.score = result.score
            await add_submission(result.submission, config.psql_db)

    logger.info(f"Final results: {task_results}")
    task.status = TaskStatus.SUCCESS

    return task
