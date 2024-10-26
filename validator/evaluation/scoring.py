from datetime import datetime
from scipy.stats import gmean
import numpy as np
from fiber.logging_utils import get_logger

from core.models.payload_models import EvaluationResult
import validator.core.constants as cts
from core.utils import download_s3_file
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Node, NodeAggregationResult, Submission
from validator.core.models import Task
from validator.core.models import MinerResults
from validator.db.sql import add_submission, get_aggregate_scores_since
from validator.db.sql import get_miners_assigned_to_task
from validator.db.sql import set_task_node_quality_score
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_get
from datetime import timedelta
import re

logger = get_logger(__name__)


def get_task_work_score(task: Task) -> int:
    hours = task.hours_to_complete
    model = task.model_id
    model_size = re.search(r'(\d+)(?=[bB])', model)
    model_size = int(model_size.group(1)) if model_size else 1
    return hours * model_size


async def scoring_aggregation(psql_db):
    logger.info('Starting to do scoring aggregation')
    a_few_days_ago = datetime.now() - timedelta(days=3)
    task_results = await get_aggregate_scores_since(a_few_days_ago, psql_db)
    node_aggregations = {}
    total_work_score = 0
    for task_res in task_results:
        task_work_score = get_task_work_score(task_res.task)
        total_work_score += task_work_score
        for node_score in task_res.node_scores:
            if node_score.node_id in node_aggregations:
                node_aggregation_result = node_aggregations[node_score.node_id]
            else:
                node_aggregation_result = NodeAggregationResult(
                    node_id=node_score.node_id,
                    work_sum=0,
                    summed_scores=0,
                    raw_scores=[]
                )
                node_aggregations[node_score.node_id] = node_aggregation_result
            if node_score.quality_score > cts.SCORE_THRESHOLD:
                node_aggregation_result.work_sum += task_work_score
            node_aggregation_result.summed_scores += node_score.quality_score - cts.SCORE_THRESHOLD
            node_aggregation_result.raw_scores.append(node_score.quality_score)

    for node_id, node_aggregation in node_aggregations.items():
        logger.info(node_id)
        logger.info(node_aggregation)
        node_aggregation.work_score = node_aggregation.work_sum / total_work_score
        node_aggregation.average_score = np.mean(node_aggregation.raw_scores)
        logger.info(f"The final scores for node {node_id} are Average Score: {node_aggregation.average_score}, Work Score: {node_aggregation.work_score} Task scores: {node_aggregation.work_sum}")


def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """Calculate weighted average of losses with more weight on test loss."""
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss

def calculate_scaled_score(weighted_loss: float, scale_factor: float) -> float:
    """Calculate score using exponential decay."""
    return np.exp(-weighted_loss * scale_factor)

def compute_adaptive_scale_factor(miner_results: list[MinerResults]) -> float:
    """
    Compute scale factor based only on finetuned submissions.
    """
    finetuned_results = [
        res for res in miner_results
        if res.is_finetune
        and not np.isnan(res.test_loss)
        and not np.isnan(res.synth_loss)
    ]

    if not finetuned_results or len(finetuned_results) == 1:
        logger.info("No finetuned results found for scale factor calculation")
        return 1.0

    weighted_losses = [
        calculate_weighted_loss(res.test_loss, res.synth_loss)
        for res in finetuned_results
    ]

    min_loss, max_loss = min(weighted_losses), max(weighted_losses)
    logger.info(f"Loss range for finetuned submissions - min: {min_loss:.4f}, max: {max_loss:.4f}")

    if min_loss == max_loss:
        logger.info("All finetuned submissions have identical losses, using default scale factor")
        return 2.0  # Default scale for identical losses

    scale = np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss)
    logger.info(f"Computed scale factor: {scale:.4f}")
    return scale

def adjust_miner_scores_to_be_relative_to_other_comps(miner_results: list[MinerResults]) -> list[MinerResults]:
    """
    Adjusts scores to have geometric mean of 1.0 for finetuned submissions only.
    """
    # Get scores only from finetuned submissions
    valid_scores = [
        res.score
        for res in miner_results
        if res.is_finetune
        and not np.isnan(res.score)
        and res.score > 0  # Ensure we only consider positive scores
    ]

    if not valid_scores:
        logger.warning("No valid finetuned submissions found for score adjustment")
        return miner_results

    logger.info(f"Adjusting scores for {len(valid_scores)} finetuned submissions")
    logger.info(f"Pre-adjustment scores: {valid_scores}")

    geometric_mean = gmean(np.array(valid_scores))

    if np.isnan(geometric_mean) or np.isinf(geometric_mean) or geometric_mean <= 0:
        logger.warning(f"Invalid geometric mean: {geometric_mean}. Scores unchanged.")
        return miner_results

    logger.info(f"Geometric mean: {geometric_mean:.4f}")

    # Only adjust scores for finetuned submissions
    for res in miner_results:
        if res.is_finetune and not np.isnan(res.score):
            original_score = res.score
            res.score /= geometric_mean
            logger.info(f"Miner {res.node_id}: {original_score:.4f} -> {res.score:.4f}")
        else:
            res.score = 0.0
            logger.info(f"Miner {res.node_id}: score set to 0.0 (non-finetuned or invalid)")

    return miner_results

def add_raw_scores_to_miner_results(miner_results: list[MinerResults]) -> list[MinerResults]:
    """
    Calculate scores using only finetuned submissions.
    Non-finetuned submissions get score of 0.0.
    """
    logger.info("Beginning score calculation...")

    # First, set all non-finetuned scores to 0.0
    for result in miner_results:
        if not result.is_finetune:
            result.score = 0.0
            logger.info(f"Miner {result.node_id}: Non-finetuned, score set to 0.0")

    # Get valid finetuned results
    finetuned_results = [
        res for res in miner_results
        if res.is_finetune
        and not np.isnan(res.test_loss)
        and not np.isnan(res.synth_loss)
    ]

    if not finetuned_results:
        logger.warning("No valid finetuned submissions found. All scores set to 0.0")
        for result in miner_results:
            result.score = 0.0
        return miner_results

    # Calculate scale factor using only finetuned submissions
    scale_factor = compute_adaptive_scale_factor(finetuned_results)
    logger.info(f"Using scale factor: {scale_factor} (calculated from {len(finetuned_results)} finetuned submissions)")

    # Calculate scores only for finetuned submissions
    for result in miner_results:
        if result.is_finetune and not np.isnan(result.test_loss) and not np.isnan(result.synth_loss):
            weighted_loss = calculate_weighted_loss(result.test_loss, result.synth_loss)
            result.score = calculate_scaled_score(weighted_loss, scale_factor)
            logger.info(
                f"Miner {result.node_id} (finetuned):"
                f" test_loss={result.test_loss:.4f}"
                f" synth_loss={result.synth_loss:.4f}"
                f" weighted_loss={weighted_loss:.4f}"
                f" score={result.score:.4f}"
            )
        else:
            result.score = 0.0
            logger.info(f"Miner {result.node_id}: score=0.0 (non-finetuned or invalid losses)")

    return miner_results

async def evaluate_and_score(task: Task, config: Config) -> Task:
    """Main evaluation and scoring function."""
    miner_pool = await get_miners_assigned_to_task(str(task.task_id), config.psql_db)
    assert task.task_id is not None
    task_results = []
    if task.instruction:
        dataset_type = CustomDatasetType(
            field_system=task.system,
            field_instruction=task.instruction,
            field_input=task.input,
            field_output=task.output
        )
    else:
        dataset_type = CustomDatasetType(
            field_system=task.system,
            field_input=task.input,
            field_instruction=task.input,
            field_output=task.output
        )

    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")

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
                logger.info(f"No submission found for miner {miner.node_id}")
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
            logger.info(f"Processing submission from miner {miner.node_id}")
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
            logger.info(f"Attempting with {evaluation_params}")

            assert task.synthetic_data is not None
            assert task.test_data is not None

            synthetic_data_filepath = await download_s3_file(task.synthetic_data)
            test_data_filepath = await download_s3_file(task.test_data)

            synth_eval_result = await run_evaluation_docker(
                dataset=synthetic_data_filepath, **evaluation_params
            )
            if synth_eval_result.is_finetune:
                test_eval_result = await run_evaluation_docker(
                    dataset=test_data_filepath, **evaluation_params
                )
            else:
                logger.info('Since the is_finetune is False, we do not need to run against the test set')
                synth_eval_result.eval_loss = 0.0
                synth_eval_result.perplexity = 0.0
                test_eval_result = EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0)


            logger.info(
                f"Evaluation results for miner {miner.node_id}:"
                f" synth_loss={synth_eval_result.eval_loss:.4f}"
                f" test_loss={test_eval_result.eval_loss:.4f}"
                f" is_finetune={test_eval_result.is_finetune}"
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
    logger.info("Beginning scoring process...")
    task_results = add_raw_scores_to_miner_results(task_results)
    task_results = adjust_miner_scores_to_be_relative_to_other_comps(task_results)

    # Update database
    logger.info("Updating database with final scores...")
    for result in task_results:
        assert result.score is not None
        await set_task_node_quality_score(task.task_id, result.node_id, result.score, config.psql_db)
        if result.submission is not None:
            result.submission.score = result.score
            await add_submission(result.submission, config.psql_db)

    logger.info("Final results:")
    for result in task_results:
        logger.info(
            f"Miner {result.node_id}:"
            f" test_loss={result.test_loss:.4f}"
            f" synth_loss={result.synth_loss:.4f}"
            f" is_finetune={result.is_finetune}"
            f" final_score={result.score:.4f}"
        )

    task.status = TaskStatus.SUCCESS
    return task
