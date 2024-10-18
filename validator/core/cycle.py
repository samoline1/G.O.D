import asyncio
import datetime
import random
from typing import List
from uuid import UUID


from core.constants import REPO_OWNER, MINIMUM_MINER_POOL
from core.models.payload_models import MinerTaskRequst, MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.utility_models import CustomDatasetType, FileFormat, TaskStatus


from validator.core.models import Node
from validator.core.models import Task
from validator.db import sql
from validator.evaluation.scoring import evaluate_and_score
from validator.tasks.task_prep import prepare_task

import validator.constants as cst
from validator.utils.call_endpoint import process_non_stream, process_stream

from fiber.logging_utils import get_logger
from datasets import get_dataset_infos

logger = get_logger(__name__)

def get_total_dataset_size(repo_name: str) -> int:
    return sum(info.dataset_size for info in get_dataset_infos(repo_name).values()
               if info.dataset_size)

async def _run_task_prep(task: Task) -> Task:
    output_task_repo_name = f"{REPO_OWNER}/{task.ds_id.replace('/', '_')}"
    columns_to_sample = [task.system, task.instruction, task.input, task.output]
    # only non-null
    columns_to_sample = list(filter(None, columns_to_sample))
    test_data, synth_data, train_data = await prepare_task(dataset_name=task.ds_id, columns_to_sample=columns_to_sample, repo_name=output_task_repo_name)
    task.hf_training_repo = train_data
    task.status =  TaskStatus.TRAINING
    task.synthetic_data = synth_data
    task.test_data = test_data
    return task


async def _make_offer(miner: Node, request: MinerTaskRequst) -> MinerTaskResponse:
    url = f"{miner.ip}:{miner.port}/task_offer/"
    return await process_non_stream(url, None, request.model_dump())


async def select_miner_pool(task: Task, miners: List[Node]):
    random.shuffle(miners)
    selected_miners = []
    num_rows_in_ds = get_total_dataset_size(task.ds_id)
    task_details_for_miner = MinerTaskRequst(
        ds_size = num_rows_in_ds,
        model = task.model_id,
        hours_to_complete = task.hours_to_complete
    )
    while len(selected_miners) < MINIMUM_MINER_POOL and miners:
        miner = miners.pop(0)
        logger.info('LOOKING FOR MINERS')
        response = await _make_offer(miner, task_details_for_miner)
        logger.info(f'The response was {response}')
        if response:
            logger.info(f'Miner {miner.node_id}  the task')
            selected_miners.append(miner.node_id)
    if len(selected_miners) < MINIMUM_MINER_POOL:
        logger.info('THERE WAS A PROBLEM - NOT ENOUGH MINERS')
        task.status = TaskStatus.FAILURE

    task.assigned_miners = selected_miners
    logger.info(f'So we have {selected_miners} assigned to the task')
    task.status = TaskStatus.MINERS_SELECTED
    return task

async def start_miners(task: Task, miners : List[Node], config):
    dataset_type = CustomDatasetType(
            field_system = task.system,
            field_input = task.input,
            field_output = task.output,
            field_instruction = task.instruction
            )

    task_request_body = TrainRequest(dataset = task.hf_training_repo,
                 model = task.model_id,
                 dataset_type= dataset_type,
                 file_format= FileFormat.S3,
                 task_id = str(task.task_id)
                 )

    for miner in miners:
        url = f"{miner.ip}:{miner.port}/start_training/"
        response = await process_stream(url, None, task_request_body.model_dump())
        logger.info(f"The response we got from {miner.node_id} was {response}")
        return response


async def process_pending_tasks(config):
    pending_tasks = await sql.get_tasks_by_status(TaskStatus.PENDING, config.psql_db)
    miner_pool = await sql.get_all_miners(config.psql_db)

    async def assign_miners(task):
        try:
            task = await select_miner_pool(task, miner_pool)
            logger.info(f"After miner assignement we have {task}")
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    await asyncio.gather(*[assign_miners(task) for task in pending_tasks[:cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]])

async def process_miner_selected_tasks(config):
    miner_selected_tasks = await sql.get_tasks_by_status(TaskStatus.MINERS_SELECTED, config.psql_db)

    async def prep_task(task):
        try:
            task = await _run_task_prep(task)
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error prepping task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    await asyncio.gather(*[prep_task(task) for task in miner_selected_tasks[:cst.MAX_CONCURRENT_TASK_PREPS]])

async def process_ready_to_train_tasks(config):
    ready_to_train_tasks = await sql.get_tasks_by_status(TaskStatus.TRAINING, config.psql_db)

    async def start_training(task):
        try:
            task.started_timestamp = datetime.datetime.now()
            task.end_timestamp = task.started_timestamp + datetime.timedelta(hours=task.hours_to_complete)
            logger.info(task)
            assigned_miners = await sql.get_miners_assigned_to_task(task.task_id, config.psql_db)
            await sql.update_task(task, config.psql_db)
            await start_miners(task, assigned_miners, config)
        except Exception as e:
            logger.error(f"Error starting training for task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    await asyncio.gather(*[start_training(task) for task in ready_to_train_tasks[:cst.MAX_CONCURRENT_TRAININGS]])

async def process_completed_tasks(config):
    completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)

    async def evaluate_task(task):
        try:
            task = await evaluate_and_score(task, config)
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    for task in completed_tasks[:cst.MAX_CONCURRENT_EVALUATIONS]:
        await evaluate_task(task)

async def validator_cycle(config):
    while True:
        try:
            logger.info("Validator Heartbeat! It's alive!")

            await process_pending_tasks(config)
            await process_miner_selected_tasks(config)
            await process_ready_to_train_tasks(config)
            await process_completed_tasks(config)

            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in validator cycle: {e}", exc_info=True)

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
