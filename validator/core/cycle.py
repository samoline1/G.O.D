import asyncio
import datetime
import random
from typing import List

from datasets import get_dataset_infos
from fiber.logging_utils import get_logger

import validator.core.constants as cst
from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Node
from validator.core.models import Task
from validator.db import sql
from validator.evaluation.scoring import evaluate_and_score
from validator.tasks.task_prep import prepare_task
from validator.utils.call_endpoint import process_non_stream


logger = get_logger(__name__)


def _get_total_dataset_size(repo_name: str) -> int:
    return sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size)


async def _run_task_prep(task: Task) -> Task:
    columns_to_sample = [i for i in [task.system, task.instruction, task.input, task.output] if i is not None]
    test_data, synth_data, train_data = await prepare_task(dataset_name=task.ds_id, columns_to_sample=columns_to_sample)
    task.hf_training_repo = train_data
    task.status = TaskStatus.READY
    task.synthetic_data = synth_data
    task.test_data = test_data
    return task


async def _make_offer(node: Node, request: MinerTaskRequst) -> MinerTaskResponse:
    url = f"{node.ip}:{node.port}/task_offer/"
    return await process_non_stream(url, None, request.model_dump())

async def _select_miner_pool_and_add_to_task(task: Task, nodes: List[Node]) -> Task:
    if len(nodes) < cst.MINIMUM_MINER_POOL:
        logger.warning(f"Not enough nodes available. Need at least {cst.MINIMUM_MINER_POOL}, but only have {len(nodes)}.")
        task.status = TaskStatus.FAILURE
        return task

    selected_miners: List[str] = []
    ds_size = _get_total_dataset_size(task.ds_id)
    task_request = MinerTaskRequest(ds_size=ds_size, model=task.model_id, hours_to_complete=task.hours_to_complete)

    # Create a copy of the nodes list to avoid mutating the original
    available_nodes = nodes.copy()

    while len(selected_miners) < cst.MINIMUM_MINER_POOL and available_nodes:
        node = random.choice(available_nodes)
        available_nodes.remove(node)

        logger.info(f"Offering node {node.node_id} the task")
        offer_response = await _make_offer(node, task_request)
        logger.info(f"Node {node.node_id}'s response to the offer was {offer_response}")

        if offer_response.accepted:
            logger.info(f"Node {node.node_id} accepted the task")
            selected_miners.append(node.node_id)

    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough miners accepted the task. We only have {len(selected_miners)} but we "
            f"need at least {cst.MINIMUM_MINER_POOL}"
        )
        task.status = TaskStatus.FAILURE
        return task

    task.assigned_miners = selected_miners
    logger.info(f"We have {len(selected_miners)} miners assigned to the task - which is enough to get going 🚀")
    task.status = TaskStatus.MINERS_SELECTED
    return task

async def _let_miners_know_to_start_training(task: Task, nodes: List[Node]):
    dataset_type = CustomDatasetType(
        field_system=task.system, field_input=task.input, field_output=task.output, field_instruction=task.instruction
    )

    task_request_body = TrainRequest(
        dataset=task.hf_training_repo,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
    )

    for node in nodes:
        url = f"{node.ip}:{node.port}/{cst.START_TRAINING_ENDPOINT}/"
        response = await process_non_stream(url, None, task_request_body.model_dump())
        logger.info(f"The response we got from {node.node_id} was {response}")

async def assign_miners(task, nodes, config):
        try:
            task = await _select_miner_pool_and_add_to_task(task, nodes)
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)


async def _process_pending_tasks(config: Config):
    pending_tasks = await sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    nodes = await sql.get_all_miners(psql_db=config.psql_db)

    await asyncio.gather(*[assign_miners(task, nodes, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]])


async def prep_task(task: Task, config: Config):
        try:
            task = await _run_task_prep(task)
            # Would much prefer you don't update the state every time and rely on that,
            # but just do the steps for each task sequentially
            ## ww - leaving this for now - something to come back to.
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error prepping task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

async def _process_miner_selected_tasks(config: Config):
    miner_selected_tasks = await sql.get_tasks_with_status(status=TaskStatus.MINERS_SELECTED, psql_db=config.psql_db)
    await asyncio.gather(*[prep_task(task, config) for task in miner_selected_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])

async def _start_training_task(task: Task, config: Config) -> None:
    try:
        task.started_timestamp = datetime.datetime.now()
        task.end_timestamp = task.started_timestamp + datetime.timedelta(hours=task.hours_to_complete)
        assigned_miners = await sql.get_miners_assigned_to_task(task.task_id, config.psql_db)
        task.status = TaskStatus.TRAINING
        await sql.update_task(task, config.psql_db)
        await _let_miners_know_to_start_training(task, assigned_miners)
    except Exception as e:
        logger.error(f"Error starting training for task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await sql.update_task(task, config.psql_db)


async def _process_ready_to_train_tasks(config: Config):
    ready_to_train_tasks = await sql.get_tasks_with_status(status=TaskStatus.READY, psql_db=config.psql_db)
    await asyncio.gather(*[_start_training_task(task, config ) for task in ready_to_train_tasks[: cst.MAX_CONCURRENT_TRAININGS]])

async def _evaluate_task(task: Task, config: Config):
        try:
            task = await evaluate_and_score(task, config)
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)


async def process_completed_tasks(config):
    completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)
    while True:
        completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)
        for task in completed_tasks:
            await _evaluate_task(task, config)
        await asyncio.sleep(5)


async def process_pending_tasks(config):
    while True:
        await _process_pending_tasks(config)
        await _process_miner_selected_tasks(config)
        await _process_ready_to_train_tasks(config)
        await asyncio.sleep(5)



  async def validator_cycle(config):
       try:
           await asyncio.gather(
               process_completed_tasks(config),
               process_pending_tasks(config),
           )
       except Exception as e:
           logger.error(f"Error in validator_cycle: {e}", exc_info=True)

async def run_validator_cycles(config):
       while True:
           try:
               await validator_cycle(config)
           except Exception as e:
               logger.error(f"Validator cycle crashed: {e}", exc_info=True)
               await asyncio.sleep(30)

def init_validator_cycles(config):
       return asyncio.create_task(run_validator_cycles(config))

