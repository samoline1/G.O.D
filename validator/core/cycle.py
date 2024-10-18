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

import json

from validator.utils.call_endpoint import process_non_stream, process_stream

from fiber.logging_utils import get_logger
logger = get_logger(__name__)

async def run_task_prep(task: Task) -> Task:
    output_task_repo_name = f"{REPO_OWNER}/{task.ds_id.replace('/', '_')}"
    columns_to_sample = [task.system, task.instruction, task.input, task.output]
    # only non-null
    columns_to_sample = list(filter(None, columns_to_sample))
    test_data, synth_data, train_data = await prepare_task(dataset_name=task.ds_id, columns_to_sample=columns_to_sample, repo_name=output_task_repo_name)
    task.hf_training_repo = train_data
    task.status =  TaskStatus.TRAINING
    task.synthetic_data = json.dumps(synth_data)
    task.test_data = json.dumps(test_data)
    return task


async def make_offer(miner: Node, request: MinerTaskRequst) -> MinerTaskResponse:
    url = f"{miner.ip}:{miner.port}/task_offer/"
    return await process_non_stream(url, None, request.model_dump())


async def select_miner_pool(task: Task, miners: List[Node]):
    random.shuffle(miners)
    selected_miners = []
    task_details_for_miner = MinerTaskRequst(
        hf_training_repo = task.hf_training_repo,
        model = task.model_id,
        hours_to_complete = task.hours_to_complete
    ) # things we give to miner to ask if they want to accept the job
    while len(selected_miners) < MINIMUM_MINER_POOL and miners:
        miner = miners.pop(0)
        logger.info('LOOKING FOR MINERS')
        response = await make_offer(miner, task_details_for_miner)
        logger.info(f'The response was {response}')
        if response:
            logger.info(f'Miner {miner.node_id}  the task')
            selected_miners.append(miner.node_id)
    if len(selected_miners) < MINIMUM_MINER_POOL:
        logger.info('THERE WAS A PROBLEM - NOT ENOUGH MINERS')
        task.status = TaskStatus.FAILURE

    # TODO: how do you want to miners to be saved into the assigned miners
    # - it's ok to pass the entire miner object? better to just give the ids?
    task.assigned_miners = selected_miners
    logger.info(f'So we have {selected_miners} assigned to the task')

    return task

async def start_miners(task: Task, miners : List[UUID], config):
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
    logger.info(f'Task is ready  for  {len(miners)} miners - lets ping them')

    for miner_id in miners:
        miner = await sql.get_node(miner_id, config.psql_db)
        url = f"{miner.ip}:{miner.port}/start_training/"
        response = await process_stream(url, None, task_request_body.model_dump())
        logger.info(f"The response we got from {miner.node_id} was {response}")
        return response


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
                logger.info(f'So now we have a task {task}')
                if task.status == TaskStatus.TRAINING:
                    logger.info(f"Asking miners to begin training!")
                    task.started_timestamp =  datetime.datetime.now()
                    task.end_timestamp = task.started_timestamp +  datetime.timedelta(hours=task.hours_to_complete)
                    await sql.update_task(task, config.psql_db)
                    await start_miners(task, task.assigned_miners, config)
            # TODO: this needs implementing
            completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)
            for completed_task in completed_tasks:
                try:
                    await evaluate_and_score(completed_task, config)
                except Exception as e:
                    logger.info(f"There was an error with validation {e}")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
