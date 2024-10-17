import os
from typing import Dict
from urllib.parse import urlparse

import aiohttp
import numpy as np
from fiber.logging_utils import get_logger

import core.constants as cts
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from validator.core.models import Task
from validator.db.sql import get_miners_assigned_to_task
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.utils.call_endpoint import process_non_stream


logger = get_logger(__name__)

from scipy.stats import gmean


async def download_s3_file(file_url: str) -> str:
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    local_file_path = os.path.join("/tmp", file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, 'wb') as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path

def calculate_relative_scores(task_scores: Dict[str, float]) -> Dict[str, float]:
    scores = np.array(list(task_scores.values()))
    geometric_mean = gmean(scores)
    relative_scores = scores / geometric_mean
    return {miner_id: float(score) for miner_id, score in zip(task_scores.keys(), relative_scores)}

async def evaluate_and_score(task: Task, config) -> Dict[str, float]:
    miner_pool = await get_miners_assigned_to_task(task.task_id, config.psql_db)
    task_results = {}
    dataset_type = CustomDatasetType(
        field_system=task.system,
        field_instruction=task.instruction,
        field_input=task.input,
        field_output=task.output
    )

    for miner in miner_pool:
        try:
            url = f"{miner.url}:{miner.port}/get_miner_latest_submission/{task.task_id}"
            submission_repo = await process_non_stream(url, None, None)
            evaluation_params = {
                'file_format': FileFormat.JSON,
                'original_model': task.model_id,
                'model': submission_repo,
                'dataset_type': dataset_type
            }

            synthetic_data_filepath = download_s3_file(task.synthetic_data)
            test_data_filepath = download_s3_file(task.test_data)

            synth_loss, is_finetune = await run_evaluation_docker(dataset=synthetic_data_filepath, **evaluation_params)
            test_loss, _ = await run_evaluation_docker(dataset=test_data_filepath, **evaluation_params)

            if is_finetune:
                weighted_loss = cts.TEST_SCORE_WEIGHTING * test_loss + (1 - cts.TEST_SCORE_WEIGHTING) * synth_loss
                task_results[miner.node_id] = 1 / weighted_loss
            else:
                task_results[miner.node_id] = 0.0

        except Exception as e:
            print(f"Error evaluating miner {miner.node_id}: {str(e)}")
            task_results[miner.node_id] = 0.0

    return calculate_relative_scores(task_results)
