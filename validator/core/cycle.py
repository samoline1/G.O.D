import asyncio
import datetime
import random
from typing import List

from loguru import logger

import validator.core.constants as cst
from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import TaskRequest
from core.models.utility_models import TaskStatus

# TODO: we shouldn't be importing these but calling the endpoint
from miner.endpoints.tuning import task_offer
from miner.endpoints.tuning import tune_model
from validator.core.models import Node
from validator.core.models import Task
from validator.db import sql
from validator.evaluation.scoring import evaluate_and_score
from validator.tasks.task_prep import prepare_task


def run_task_prep(task: Task) -> Task:
    output_task_repo_name = f'{cst.REPO_OWNER}/{task.id}'
    test_data, synth_data = prepare_task(
        dataset_name=task.repo_name,
        columns_to_sample=[
            task.system,
            task.instruction,
            task.input_data,
            task.output
        ],
        repo_name=output_task_repo_name
    )
    task.status =  TaskStatus.TRAINING
    task.synthetic_data = synth_data
    task.test_data = test_data
    return task

def select_miner_pool(task: Task, miners: List[Node]):
    random.shuffle(miners)
    selected_miners = []
    task_details_for_miner = MinerTaskRequst(
        hf_training_repo = task.hf_training_repo,
        model = task.model_id,
        hours_to_complete = task.hours_to_complete
    ) # things we give to miner to ask if they want to accept the job
    while len(selected_miners) < task.num_miners_required and miners:
        miner = miners.pop(0)
        # TODO: right now I just call the miner function, need to instead call the miner api
        if task_offer(task_details_for_miner):
            selected_miners.append(miner.node_id)
    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        task.status = TaskStatus.FAILURE

    # TODO: how do you want to miners to be saved into the assigned miners
    # - it's ok to pass the entire miner object? better to just give the ids?
    task.assigned_miners = selected_miners

    return task

async def start_miners(task: Task, miners : List[Node]):
    task_request_body = TaskRequest(ds_repo = task.hf_training_repo,
                                    system_col = task.system,
                                    input_col = task.input,
                                    instruction_col = task.instruction,
                                    output_col = task.output,
                                    model_repo = task.model_id,
                                    hours_to_complete = task.hours_to_complete
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
                task = run_task_prep(task)
                miner_pool = await sql.get_all_miners(config.psql_db)
                task = select_miner_pool(task, miner_pool)
                if task.status == TaskStatus.TRAINING:
                    task.started_timestamp =  datetime.datetime.now()
                    await start_miners(task, miner_pool)
            # TODO: this needs implementing
            completed_tasks = await sql.get_tasks_by_status('complete', config.psql_db)
            for completed_task in completed_tasks:
                await evaluate_and_score(completed_task, config)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
