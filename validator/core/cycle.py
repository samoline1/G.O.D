import asyncio
import datetime
import random
from typing import List

from fiber.logging_utils import get_logger

import validator.core.constants as cst
from core.constants import REPO_OWNER, MINIMUM_MINER_POOL
from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import TrainRequest
from core.models.utility_models import CustomDatasetType, FileFormat, TaskStatus


# TODO: we shouldn't be importing these but calling the endpoint
from miner.endpoints.tuning import task_offer
from miner.endpoints.tuning import tune_model
from validator.core.models import Node
from validator.core.models import Task
from validator.db import sql
from validator.evaluation.scoring import evaluate_and_score
from validator.tasks.task_prep import prepare_task

import json
logger = get_logger(__name__)

NUM_MINERS_REQUIRED = 1 # dev temp
async def run_task_prep(task: Task) -> Task:
    output_task_repo_name = f"{REPO_OWNER}/{task.ds_id.replace('/', '_')}"
    columns_to_sample = [task.system, task.instruction, task.input, task.output]
    # only non-null
    columns_to_sample = list(filter(None, columns_to_sample))
    test_data, synth_data = await prepare_task(dataset_name=task.ds_id, columns_to_sample=columns_to_sample, repo_name=output_task_repo_name)
    task.hf_training_repo = output_task_repo_name
    task.status =  TaskStatus.TRAINING
    task.synthetic_data = json.dumps(synth_data)
    task.test_data = json.dumps(test_data)
    return task

async def select_miner_pool(task: Task, miners: List[Node]):
    random.shuffle(miners)
    selected_miners = []
    task_details_for_miner = MinerTaskRequst(
        hf_training_repo = task.hf_training_repo,
        model = task.model_id,
        hours_to_complete = task.hours_to_complete
    ) # things we give to miner to ask if they want to accept the job
    while len(selected_miners) < MINIMUM_MINER_POOL and miners:
        miner = miners.pop(1)
        # TODO: right now I just call the miner function, need to instead call the miner api
        logger.info('LOOKING FOR MINERS')
        response = await task_offer(task_details_for_miner)
        if response:
            logger.info(f'Miner {miner.node_id}  the task')
            selected_miners.append(miner.node_id)
    if len(selected_miners) < MINIMUM_MINER_POOL:
        logger.info('THERE WAS A PROBLEM - NOT ENOUGH MINERS')
        task.status = TaskStatus.FAILURE

    # TODO: how do you want to miners to be saved into the assigned miners
    # - it's ok to pass the entire miner object? better to just give the ids?
    task.assigned_miners = selected_miners

    return task

async def start_miners(task: Task, miners : List[Node]):
    dataset_type = CustomDatasetType(
            field_system = task.system,
            field_input = task.input,
            field_output = task.output,
            field_instruction = task.instruction
            )

    task_request_body = TrainRequest(dataset = task.hf_training_repo,
                 model = task.model_id,
                 dataset_type= dataset_type,
                 file_format= FileFormat.HF,
                 task_id = task.task_id
                 )
    for miner in miners:
        # TODO: rather than calling the function directly, we should be calling the endpoint given the miner url etc
        await tune_model(task_request_body) # ( will return true when started so doesn't matter we are awaiting for testing)


async def validator_cycle(config):
    try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            tasks = await sql.get_tasks_by_status(TaskStatus.PENDING, config.psql_db)
            for task in tasks:
                task = await run_task_prep(task)
                miner_pool = await sql.get_all_miners(config.psql_db)
                task = await select_miner_pool(task, miner_pool)
                await sql.update_task(task, config.psql_db)
                if task.status == TaskStatus.TRAINING:
                    task.started_timestamp =  datetime.datetime.now()
                    await start_miners(task, miner_pool)
            # TODO: this needs implementing
            completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)
            for completed_task in completed_tasks:
                await evaluate_and_score(completed_task, config)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
