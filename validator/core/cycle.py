import asyncio
from typing import List
import random
import datetime
from loguru import logger

from validator.db import sql
from validator.tasks.task_prep import prepare_task
from validator.core.models import Task, Node
import validator.core.constants as cst
from core.models.payload_models import MinerTaskRequst
from core.models.utility_models import TaskStatus
import core.constants as cst
from validator.evaluation.scoring import evaluate_and_score

def run_task_prep(task: Task) -> Task:
    output_task_repo_name = task.id
    test_data, synth_data = prepare_task(dataset_name=task.repo_name, columns_to_sample=[task.system, task.instruction, task.input_data, task.output], repo_name=output_task_repo_name)
    task.status =  TaskStatus.TRAINING
    return task


def select_miner_pool(task: Task, miners: List[Node]):
    random.shuffle(miners)
    selected_miners = []
    task_details_for_miner = MinerTaskRequst(hf_training_repo = task.hf_training_repo, model = task.model_id) # things we give to miner to ask if they want to accept the job
    while len(selected_miners) < task.num_miners_required and miners:
        miner = miners.pop(0)
        # TODO: implement the below
        if miner_accepts(task_details_for_miner):
            selected_miners.append(miner.node_id)
    # How will this be added to the ds?  As the IDs right?
    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        task.status = TaskStatus.FAILURE
    task.assigned_miners = selected_miners

    return task



async def validator_cycle(config):
     try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            tasks = await sql.get_tasks_by_status(TaskStatus.PENDING, config.psql_db)
            for task in tasks:
                task = run_task_prep(task)
                miner_pool = sql.get_all_miners(config.psql_db)
                task = select_miner_pool(task, miner_pool)
                if task.status == TaskStatus.TRAINING:
                    task.started_timestamp =  datetime.datetime.now()
            completed_tasks = await sql.get_tasks_ready_for_evaluation(config.psql_db)
            for completed_task in completed_tasks:
                await evaluate_and_score(completed_task, config)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
