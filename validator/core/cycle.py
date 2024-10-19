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
from validator.utils.call_endpoint import process_stream


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


# needs to be a lot more white space in this function, you can't read it because it's all jammed together
async def _select_miner_pool(task: Task, nodes: List[Node]) -> Task:  # typehint?
    nodes = random.sample(nodes, k=cst.MINIMUM_MINER_POOL)

    selected_miners = []  # typehint?
    ds_size = _get_total_dataset_size(task.ds_id)
    task_request = MinerTaskRequst(ds_size=ds_size, model=task.model_id, hours_to_complete=task.hours_to_complete)

    while len(nodes) > 0:
        node = nodes.pop()  # Popping at the start is a bad idea, you're mutating the list
        # This means it is O(n) for every operation, meaning the whole process is O(n^2) instead of O(n)
        logger.info(f"Offering node {node.node_id} the task")
        offer_response = await _make_offer(node, task_request)
        logger.info(f"Node {node.node_id}'s response to the offer was {offer_response}")

        # Before you wasn't even checking the accepted bool - this would've flagged yes to everything
        # This is why in general `if X` is something to avoid
        if offer_response.accepted is True:
            logger.info(f"Node {node.node_id} accepted the task")
            selected_miners.append(node.node_id)

    # NOTE: this is also very bugged. You don't return after this check, so you'll keep going always anyway
    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        # If there is a problem, warn or error
        logger.warning(
            f"Hmm, we didn't get enough miners to do the task. We only have {len(selected_miners)} but we"
            f"need at least {cst.MINIMUM_MINER_POOL}"
        )
        task.status = TaskStatus.FAILURE
        return task

    task.assigned_miners = selected_miners
    logger.info(f"We have {selected_miners} assigned to the task - which is enough to get going 🚀")
    task.status = TaskStatus.MINERS_SELECTED
    return task

# better name plz
async def _start_miners(task: Task, nodes: List[Node]):
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

    # NOTE: what is this doing? For each miner sending a streamed request for training? what
    # Why is it a streamed response in openai like schemas?
    for node in nodes:
        # nO magic strings, make this constant plz
        url = f"{node.ip}:{node.port}/start_training/"
        # Token is typehinted as string but your'e passing None
        response = await process_stream(url, None, task_request_body.model_dump())
        logger.info(f"The response we got from {node.node_id} was {response}")
        return response


async def _process_pending_tasks(config: Config):
    pending_tasks = await sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    nodes = await sql.get_all_miners(psql_db=config.psql_db)  # this isn't the miner pool, this is just nodes

    # nOTE: why is this herE?
    async def assign_miners(task):
        try:
            # We've said _select_miner_pool, then assigned the result to task?#
            # Wouldn't I expect to get the miner pool back?
            task = await _select_miner_pool(task, nodes)
            # What happens if there was not enough miners?
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    await asyncio.gather(*[assign_miners(task) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]])


async def _process_miner_selected_tasks(config: Config):
    miner_selected_tasks = await sql.get_tasks_with_status(status=TaskStatus.MINERS_SELECTED, psql_db=config.psql_db)
    # no inner funcs pls
    async def prep_task(task):
        # you use this try except block everywhere
        # TODO: use a func or context manager or something to do this everywhehre perhaps?
        try:
            task = await _run_task_prep(task)
            # Would much prefer you don't update the state every time and rely on that,
            # but just do the steps for each task sequentially
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error prepping task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    await asyncio.gather(*[prep_task(task) for task in miner_selected_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])

# my eyes bleed, where are the typehints
# Also why inner function?
# Also name of this should be more like _start_training_task - we aren't training anything in the vali
async def _start_training(task: Task, config: Config) -> None:
    try:
        task.started_timestamp = datetime.datetime.now()
        task.end_timestamp = task.started_timestamp + datetime.timedelta(hours=task.hours_to_complete)
        assigned_miners = await sql.get_miners_assigned_to_task(task.task_id, config.psql_db)
        task.status = TaskStatus.TRAINING
        await sql.update_task(task, config.psql_db)
        await _start_miners(task, assigned_miners)
    except Exception as e:
        logger.error(f"Error starting training for task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await sql.update_task(task, config.psql_db)


async def _process_ready_to_train_tasks(config: Config):
    ready_to_train_tasks = await sql.get_tasks_with_status(status=TaskStatus.READY, psql_db=config.psql_db)

    await asyncio.gather(*[_start_training(task, config ) for task in ready_to_train_tasks[: cst.MAX_CONCURRENT_TRAININGS]])


async def process_completed_tasks(config):
    completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)

    # no need for inner function - inherits state and is redefined every time
    async def evaluate_task(task):
        try:
            task = await evaluate_and_score(task, config)
            await sql.update_task(task, config.psql_db)
        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
            task.status = TaskStatus.FAILURE
            await sql.update_task(task, config.psql_db)

    for task in completed_tasks[: cst.MAX_CONCURRENT_EVALUATIONS]:
        await evaluate_task(task)


# FOR demonstartion, please finish

async def evaluate_completed_tasks(config):
    while True:
        completed_tasks = await sql.get_tasks_ready_to_evaluate(config.psql_db)
        for task in completed_tasks[: cst.MAX_CONCURRENT_EVALUATIONS]:
            await _evaluate_task(task, config)  # noqa
        await asyncio.sleep(5)


async def process_pending_tasks(config):
    while True:
        tasks = await _process_pending_tasks(config)
        tasks = await _process_miner_selected_tasks(config, tasks)
        await _process_ready_to_train_tasks(config, tasks)
        await asyncio.sleep(5)


# NOTE: biggest change is here - in its old form it would not work
# As you won't be able to evaluate and send out new tasks
async def validator_cycle(config):
    await asyncio.gather(
        evaluate_completed_tasks(config),
        process_pending_tasks(config),
    )


# So this is really dangerous because you have a background task that is NEVER awaited
# so if it errors, it will just quietly die and not bubble errors up the surface
def init_validator_cycles(config):
    return asyncio.create_task(validator_cycle(config))
