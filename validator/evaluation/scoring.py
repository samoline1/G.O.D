from datetime import datetime, timedelta
from scipy.stats import gmean
import numpy as np
from fiber.logging_utils import get_logger

from core.models.payload_models import EvaluationResult
import validator.core.constants as cts
from core.utils import download_s3_file
from core.models.utility_models import CustomDatasetType, FileFormat, TaskStatus
from validator.core.config import Config
from validator.core.models import Node, NodeAggregationResult, Submission, TaskNode
from validator.core.models import Task, TaskResults
from validator.core.models import MinerResults
from validator.db.sql import add_submission, get_aggregate_scores_since
from validator.db.sql import get_miners_assigned_to_task
from validator.db.sql import set_task_node_quality_score
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream_get
import re

logger = get_logger(__name__)

def get_task_work_score(task: Task) -> float:
    """Calculate work score for a task based on hours and model size."""
    assert task.hours_to_complete > 0, "Hours to complete must be positive"
    assert task.model_id, "Model ID must be present"

    hours = task.hours_to_complete
    model = task.model_id
    model_size = re.search(r'(\d+)(?=[bB])', model)
    model_size_value = int(model_size.group(1)) if model_size else 1

    logger.info(f"Task_id {task.task_id} start_time: {task.started_timestamp} model: {model} size {model_size_value} hours: {hours} data: {task.ds_id}")
    return np.log(float(hours * model_size_value))

def calculate_adjusted_task_score(quality_score: float, task_work_score: float) -> float:
    """Calculate adjusted task score based on quality score and work score."""
    assert not np.isnan(quality_score), "Quality score cannot be NaN"
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"
    return max(cts.MIN_TASK_SCORE, quality_score - cts.TASK_SCORE_THRESHOLD) * task_work_score

def update_node_aggregation(
    node_aggregations: dict[int, NodeAggregationResult],
    node_score: TaskNode,
    task_work_score: float
) -> None:
    """Update node aggregation results with new scores for a particular task."""
    assert isinstance(node_score.node_id, int), "Node ID must be an integer"
    assert not np.isnan(task_work_score), "Task work score cannot be NaN"

    if node_score.node_id not in node_aggregations:
        node_aggregations[node_score.node_id] = NodeAggregationResult(node_id=node_score.node_id)

    node_result = node_aggregations[node_score.node_id]
    adjusted_score = calculate_adjusted_task_score(node_score.quality_score, task_work_score)

    node_result.summed_adjusted_task_scores += adjusted_score
    node_result.task_raw_scores.append(node_score.quality_score)
    node_result.task_work_scores.append(task_work_score)

def calculate_node_quality_scores(
    node_aggregations: dict[int, NodeAggregationResult]
) -> tuple[list[tuple[int, float]], float]:
    """Calculate quality scores for each node."""
    assert node_aggregations, "Node aggregations dictionary cannot be empty"

    final_scores: list[tuple[int, float]] = []
    min_score = float('inf')

    for node_id, node_agg in node_aggregations.items():
        assert node_agg.task_raw_scores, f"No raw scores available for node {node_id}"

        node_agg.average_raw_score = float(np.mean(node_agg.task_raw_scores))
        score = node_agg.summed_adjusted_task_scores * node_agg.average_raw_score
        node_agg.quality_score = score
        min_score = min(min_score, score)
        final_scores.append((node_id, score))

    return final_scores, min_score

def normalise_scores(
    final_scores: list[tuple[int, float]],
    min_score: float,
    node_aggregations: dict[int, NodeAggregationResult]
) -> None:
    """Normalize scores and update node emission values."""
    assert final_scores, "Final scores list cannot be empty"

    shift = abs(min_score) + 1e-10 if min_score < 0 else 0
    total = sum(score + shift for _, score in final_scores)

    for node_id, score in final_scores:
        normalized_score = (score + shift) / total if total > 0 else 1.0 / len(final_scores)
        node_aggregations[node_id].emission = normalized_score
        logger.info(str(node_aggregations[node_id]))

async def scoring_aggregation(psql_db: str) -> None:
    """Aggregate and normalize scores across all nodes."""
    try:
        a_few_days_ago = datetime.now() - timedelta(days=3)
        task_results: list[TaskResults] = await get_aggregate_scores_since(a_few_days_ago, psql_db)
        assert task_results, "No task results found"

        node_aggregations: dict[int, NodeAggregationResult] = {}

        for task_res in task_results:
            task_work_score = get_task_work_score(task_res.task)
            for node_score in task_res.node_scores:
                update_node_aggregation(node_aggregations, node_score, task_work_score)

        final_scores, min_score = calculate_node_quality_scores(node_aggregations)
        normalise_scores(final_scores, min_score, node_aggregations)

    except Exception as e:
        logger.error(f"Error in scoring aggregation: {e}")
        raise

def calculate_weighted_loss(test_loss: float, synth_loss: float) -> float:
    """Calculate weighted average of losses with more weight on test loss."""
    assert not np.isnan(test_loss), "Test loss cannot be NaN"
    assert not np.isnan(synth_loss), "Synthetic loss cannot be NaN"
    return cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss

def calculate_scaled_score(weighted_loss: float, scale_factor: float) -> float:
    """Calculate score using exponential decay."""
    assert not np.isnan(weighted_loss), "Weighted loss cannot be NaN"
    assert scale_factor > 0, "Scale factor must be positive"
    return float(np.exp(-weighted_loss * scale_factor))

def compute_adaptive_scale_factor(miner_results: list[MinerResults]) -> float:
    """Compute scale factor based only on finetuned submissions."""
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
        return 2.0

    scale = float(np.log(cts.TARGET_SCORE_RATIO) / (max_loss - min_loss))
    logger.info(f"Computed scale factor: {scale:.4f}")
    return scale

def adjust_miner_scores_to_be_relative_to_other_comps(miner_results: list[MinerResults]) -> list[MinerResults]:
    """Adjusts scores to have geometric mean of 1.0 for finetuned submissions only."""
    valid_scores = [
        res.score for res in miner_results
        if res.is_finetune
        and res.score is not None
        and not np.isnan(res.score)
        and res.score > 0
    ]

    if not valid_scores:
        logger.warning("No valid finetuned submissions found for score adjustment")
        return miner_results

    logger.info(f"Adjusting scores for {len(valid_scores)} finetuned submissions")
    logger.info(f"Pre-adjustment scores: {valid_scores}")

    geometric_mean = float(gmean(np.array(valid_scores)))

    if np.isnan(geometric_mean) or np.isinf(geometric_mean) or geometric_mean <= 0:
        logger.warning(f"Invalid geometric mean: {geometric_mean}. Scores unchanged.")
        return miner_results

    logger.info(f"Geometric mean: {geometric_mean:.4f}")

    for res in miner_results:
        if res.is_finetune and res.score is not None and not np.isnan(res.score):
            original_score = res.score
            res.score = float(res.score / geometric_mean)
            logger.info(f"Miner {res.node_id}: {original_score:.4f} -> {res.score:.4f}")
        else:
            res.score = 0.0
            logger.info(f"Miner {res.node_id}: score set to 0.0 (non-finetuned or invalid)")

    return miner_results

def add_raw_scores_to_miner_results(miner_results: list[MinerResults]) -> list[MinerResults]:
    """Calculate scores using only finetuned submissions."""
    logger.info("Beginning score calculation...")

    for result in miner_results:
        if not result.is_finetune:
            result.score = 0.0
            logger.info(f"Miner {result.node_id}: Non-finetuned, score set to 0.0")

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

    scale_factor = compute_adaptive_scale_factor(finetuned_results)
    logger.info(f"Using scale factor: {scale_factor} (calculated from {len(finetuned_results)} finetuned submissions)")

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

def _get_dataset_type(task: Task) -> CustomDatasetType:
    return CustomDatasetType(
        field_system=task.system,
        field_instruction=task.instruction or task.input,
        field_input=task.input,
        field_output=task.output
    )

def _create_failed_miner_result(node_id: int) -> MinerResults:
    return MinerResults(
        node_id=node_id,
        test_loss=np.nan,
        synth_loss=np.nan,
        is_finetune=False,
        submission=None
    )

async def _get_submission_repo(miner: Node, task_id: str) -> str | None:
    url = f"{miner.ip}:{miner.port}/get_latest_model_submission/{task_id}"
    try:
        return str(await process_non_stream_get(url, None))
    except Exception as e:
        logger.error(f"Failed to get submission for miner {miner.node_id}: {e}")
        return None

async def _evaluate_submission(
    task: Task,
    submission_repo: str,
    dataset_type: CustomDatasetType
) -> tuple[EvaluationResult, EvaluationResult]:
    evaluation_params = {
        "file_format": FileFormat.JSON,
        "original_model": task.model_id,
        "model": submission_repo,
        "dataset_type": dataset_type,
    }

    assert task.synthetic_data is not None,  "Synthetic data shouldn't be none"
    assert task.test_data is not None,  "Test data shouldn't be none"
    synthetic_data_filepath = await download_s3_file(task.synthetic_data)
    synth_eval_result = await run_evaluation_docker(
        dataset=synthetic_data_filepath, **evaluation_params
    )

    if not synth_eval_result.is_finetune:
        return (
            EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0),
            EvaluationResult(is_finetune=False, eval_loss=0.0, perplexity=0.0)
        )

    test_data_filepath = await download_s3_file(task.test_data)
    test_eval_result = await run_evaluation_docker(
        dataset=test_data_filepath, **evaluation_params
    )

    return synth_eval_result, test_eval_result

async def _process_miner(
    miner: Node,
    task: Task,
    dataset_type: CustomDatasetType
) -> MinerResults:
    assert task.task_id is not None, "We should have a task id when processing the miner"
    submission_repo = await _get_submission_repo(miner, str(task.task_id))
    if not submission_repo:
        return _create_failed_miner_result(miner.node_id)

    try:
        submission = Submission(
            task_id=task.task_id,
            node_id=miner.node_id,
            repo=submission_repo,
            created_on=datetime.now(),
            updated_on=datetime.now(),
        )

        synth_result, test_result = await _evaluate_submission(task, submission_repo, dataset_type)

        return MinerResults(
            node_id=miner.node_id,
            test_loss=float(test_result.eval_loss),
            synth_loss=float(synth_result.eval_loss),
            is_finetune=test_result.is_finetune,
            submission=submission
        )
    except Exception as e:
        logger.error(f"Error evaluating miner {miner.node_id}: {e}")
        return _create_failed_miner_result(miner.node_id)

async def _update_scores(task: Task, task_results: list[MinerResults], psql_db) -> None:
    assert task.task_id is not None, 'task id needs to be seet to update scores'
    for result in task_results:
        if result.score is None:
            continue

        await set_task_node_quality_score(
            task_id=task.task_id,
            node_id=result.node_id,
            quality_score=float(result.score),
            psql_db=psql_db
        )

        if result.submission:
            result.submission.score = result.score
            await add_submission(result.submission, psql_db)

async def evaluate_and_score(task: Task, config: Config) -> Task:
    assert task.task_id is not None, "Task ID must be present"
    assert task.synthetic_data is not None, "Synthetic data must be present"
    assert task.test_data is not None, "Test data must be present"

    miner_pool = await get_miners_assigned_to_task(str(task.task_id), config.psql_db)
    dataset_type = _get_dataset_type(task)

    logger.info(f"Beginning evaluation for task {task.task_id} with {len(miner_pool)} miners")
    task_results = [
        await _process_miner(miner, task, dataset_type)
        for miner in miner_pool
    ]

    logger.info("Calculating final scores...")
    task_results = add_raw_scores_to_miner_results(task_results)
    task_results = adjust_miner_scores_to_be_relative_to_other_comps(task_results)

    await _update_scores(task, task_results, config.psql_db)
    task.status = TaskStatus.SUCCESS
    return task
