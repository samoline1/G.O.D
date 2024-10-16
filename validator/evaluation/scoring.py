from core.models.utility_models import CustomDatasetType, FileFormat
from typing import List, Tuple, Dict
from collections import defaultdict
from validator.db.sql import get_miner_latest_submission, get_miners_assigned_to_task
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.core.models import Task
import core.constants as cts
import numpy as np

from validator.utils.call_endpoint import process_non_stream

from fiber.logging_utils import get_logger
logger = get_logger(__name__)


#def calculate_linear_score(rank: int, total_miners: int) -> float:
#    cutoff = int(total_miners * cts.TOP_SYNTH_PERCENT_CUTOFF) # if you have a synth loss > some percent% then we want to penalise you to avoid the issue of overfitting to the test set
#    if rank >= cutoff:
#        return 0
#    return 1 - cts.LOWEST_SCORE_FOR_TOP_MINERS * (rank / cutoff)

#def calculate_work_score(task: Task):
#    return task.hours_to_complete / cts.MAX_COMPETITION_HOURS

#def calculate_score(test_results: List[Tuple[str, float]], synth_results: List[Tuple[str, float]], task: Task) -> Dict[str, float]:
#    work_score = task.hours_to_complete
#
#    num_miners = len(test_results)
#
#    synth_ranks = {miner_id: rank for rank, (miner_id, _) in enumerate(synth_results)}
#
#    scores = {}
#    for test_rank, (miner_id, _) in enumerate(test_results):
#        test_score = calculate_linear_score(test_rank, num_miners)
#        synth_score = calculate_linear_score(synth_ranks[miner_id], num_miners)
#
#        # Penalize high loss synthetic performers, you test score will drop by some factor
#        if synth_ranks[miner_id] == 0:
#            test_score *= cts.PENALISATION_FACTOR_FOR_HIGH_SYNTH_LOSS
#
#        score = (cts.TEST_SCORE_WEIGHTING * test_score + (1 - cts.TEST_SCORE_WEIGHTING) * synth_score) * work_score
#        scores[miner_id] = score
#
#    return scores


async def evaluate_and_score(task: Task, config) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    miner_pool = await get_miners_assigned_to_task(task.task_id, config.psql_db)
    task_results = defaultdict(dict)
#    test_results = []
#    synth_results = []

    dataset_type = CustomDatasetType(
        field_system=task.system,
        field_instruction=task.instruction,
        field_input=task.input,
        field_output=task.output
    )

    for miner in miner_pool:
        url = f"{miner.url}:{miner.port}/get_miner_latest_submission/{task.task_id}"
        submission_repo = await process_non_stream(url, None, None)

        evaluation_params = {
                'file_format': FileFormat.JSON,
                'original_model': task.model_id,
                'model': submission_repo,
                'dataset_type': dataset_type
                }

        # TODO: rn we're adding the actual dataset to this call which is fine, but what will need to happen inside the run_evaluation_docker, if we are passing in the actual file
        # is to create storage and save the data to that location in the container
        synth_loss, is_finetune = run_evaluation_docker(dataset=task.synthetic_data, **evaluation_params)
        test_loss, _ = run_evaluation_docker(dataset=task.test_data, **evaluation_params)

        task_results[miner.node_id] = {
            'synth_loss': synth_loss if is_finetune else np.inf,
            'test_loss': test_loss if is_finetune else np.inf,
        }
        task_results[miner.node_id]['quality'] = cts.TEST_SCORE_WEIGHTING * task_results[miner.node_id]['test_loss'] + (1 - cts.TEST_SCORE_WEIGHTING) * task_results[miner.node_id]['synth_loss']


    logger.info(f"The results for this game are {task_results}")

#        test_results.append((miner.id, test_loss))
#        synth_results.append((miner.id, synth_loss))
#
#    ordered_test_results = sorted(test_results, key=lambda x: x[1])
#    ordered_synth_results = sorted(synth_results, key=lambda x: x[1])
#    scores = calculate_score(ordered_test_results, ordered_synth_results, task)

    # now we need to update miners quality scores ...

