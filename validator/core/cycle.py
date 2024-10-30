import asyncio
import datetime
import random
from datasets import get_dataset_infos
from fiber.logging_utils import get_logger
from fsspec.spec import conf

import validator.core.constants as cst
from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.models import Task
from fiber.networking.models import NodeWithFernet as Node
from validator.core.refresh_nodes import get_and_store_nodes, perform_handshakes
import validator.db.sql.tasks as tasks_sql
import validator.db.sql.nodes as nodes_sql
from validator.evaluation.scoring import evaluate_and_score, scoring_aggregation_from_date
from validator.tasks.task_prep import prepare_task
from validator.utils.call_endpoint import process_non_stream_fiber



logger = get_logger(__name__)


def _get_total_dataset_size(repo_name: str) -> int:
    return sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size)


async def _run_task_prep(task: Task) -> Task:
    columns_to_sample = [i for i in [task.system, task.instruction, task.input, task.output] if i is not None]
    test_data, synth_data, train_data = await prepare_task(dataset_name=task.ds_id, columns_to_sample=columns_to_sample)
    task.hf_training_repo = train_data
    task.status = TaskStatus.DATA_READY
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info('Data creation is complete - now time to find some miners')
    return task



async def _make_offer(node: Node, request: MinerTaskRequst, config: Config) -> MinerTaskResponse:
    logger.info(f"We are making the following offer {request.model_dump()}")
    response = await process_non_stream_fiber(cst.TASK_OFFER_ENDPOINT, config, node, request.model_dump())
    return MinerTaskResponse(message=response.get('message', 'No message given'), accepted=response.get('accepted', False))

async def _select_miner_pool_and_add_to_task(task: Task, nodes: list[Node], config: Config) -> Task:
    if len(nodes) < cst.MINIMUM_MINER_POOL:
        logger.warning(f"Not enough nodes available. Need at least {cst.MINIMUM_MINER_POOL}, but only have {len(nodes)}.")
        task.status = TaskStatus.FAILURE
        return task

    selected_miners: list[int] = []
    ds_size = _get_total_dataset_size(task.ds_id)
    task_request = MinerTaskRequst(ds_size=ds_size, model=task.model_id, hours_to_complete=task.hours_to_complete, task_id= str(task.task_id))

    # Create a copy of the nodes list to avoid mutating the original - better than popping? Not sure
    available_nodes = nodes.copy()

    num_of_miners_to_try_for = random.randint(cst.MIN_IDEAL_NUM_MINERS_IN_POOL,cst.MAX_IDEAL_NUM_MINERS_IN_POOL)
    while len(selected_miners) < num_of_miners_to_try_for and available_nodes:
        node = random.choice(available_nodes)
        available_nodes.remove(node)

        try:
            offer_response = await _make_offer(node, task_request, config)
            logger.info(f"Node {node.node_id}'s response to the offer was {offer_response}")
        except:
            logger.info(f"Seems that {node.node_id} has a connection issue")
            offer_response = MinerTaskResponse(accepted=False, message="Connection error")

        if offer_response.accepted is True:
            selected_miners.append(node.node_id)
            await tasks_sql.assign_node_to_task(str(task.task_id), node.node_id, config.psql_db)
            logger.info(f"The miner {node.node_id} has officially been assigned the task")

    if len(selected_miners) < cst.MINIMUM_MINER_POOL:
        logger.warning(
            f"Not enough miners accepted the task. We only have {len(selected_miners)} but we "
            f"need at least {cst.MINIMUM_MINER_POOL}"
        )
        task.status = TaskStatus.FAILURE
        return task

    task.assigned_miners = selected_miners
    logger.info(f"We have {len(selected_miners)} miners assigned to the task - which is enough to get going 🚀")
    task.status = TaskStatus.READY
    return task

async def _let_miners_know_to_start_training(task: Task, nodes: list[Node], config: Config):
    dataset_type = CustomDatasetType(
        field_system=task.system, field_input=task.input, field_output=task.output, field_instruction=task.instruction
    )

    dataset = task.hf_training_repo if task.hf_training_repo else "dataset error"
    task_request_body = TrainRequest(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete = task.hours_to_complete
    )
    logger.info(f'We are tellingminers to start training there are  {len(nodes)}')

    for node in nodes:
        response = await process_non_stream_fiber(cst.START_TRAINING_ENDPOINT, config, node, task_request_body.model_dump())
        logger.info(f"The response we got from {node.node_id} was {response}")

async def assign_miners(task: Task, nodes: list[Node], config: Config):
    try:
        task = await _select_miner_pool_and_add_to_task(task, nodes, config)
        await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(f"Error assigning miners to task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await tasks_sql.update_task(task, config.psql_db)


async def _find_miners_for_task(config: Config):
    pending_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.DATA_READY, psql_db=config.psql_db)
    nodes = await nodes_sql.get_all_nodes(psql_db=config.psql_db)
    await asyncio.gather(*[assign_miners(task, nodes, config) for task in pending_tasks[: cst.MAX_CONCURRENT_MINER_ASSIGNMENTS]])


async def prep_task(task: Task, config: Config):
    logger.info('PREPING TASK')
    try:
        task = await _run_task_prep(task)
        await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(f"Error prepping task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await tasks_sql.update_task(task, config.psql_db)

async def _process_selected_tasks(config: Config):
    miner_selected_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.PENDING, psql_db=config.psql_db)
    await asyncio.gather(*[prep_task(task, config) for task in miner_selected_tasks[: cst.MAX_CONCURRENT_TASK_PREPS]])

async def _start_training_task(task: Task, config: Config) -> None:
    try:
        task.started_timestamp = datetime.datetime.now()
        task.end_timestamp = task.started_timestamp + datetime.timedelta(hours=task.hours_to_complete)
        assigned_miners = await tasks_sql.get_nodes_assigned_to_task(str(task.task_id), config.psql_db)
        assigned_miners = await perform_handshakes(assigned_miners, config)
        await _let_miners_know_to_start_training(task, assigned_miners, config)
        task.status = TaskStatus.TRAINING
        await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
        logger.error(f"Error starting training for task {task.task_id}: {e}", exc_info=True)
        task.status = TaskStatus.FAILURE
        await tasks_sql.update_task(task, config.psql_db)


async def _process_ready_to_train_tasks(config: Config):
    ready_to_train_tasks = await tasks_sql.get_tasks_with_status(status=TaskStatus.READY, psql_db=config.psql_db)
    if len(ready_to_train_tasks) > 0:
        logger.info(f"There are {len(ready_to_train_tasks)} ready to train")
        await asyncio.gather(*[_start_training_task(task, config ) for task in ready_to_train_tasks[: cst.MAX_CONCURRENT_TRAININGS]])
    else:
        logger.info(f"No pending tasks - waiting for 30 seconds")
        await asyncio.sleep(30)

async def _evaluate_task(task: Task, config: Config):
    try:
       task = await evaluate_and_score(task, config)
       await tasks_sql.update_task(task, config.psql_db)
    except Exception as e:
       logger.error(f"Error evaluating task {task.task_id}: {e}", exc_info=True)
       task.status = TaskStatus.FAILURE
       await tasks_sql.update_task(task, config.psql_db)


async def process_completed_tasks(config: Config) -> None:
    while True:
        completed_tasks = await tasks_sql.get_tasks_ready_to_evaluate(config.psql_db)
        if len(completed_tasks) > 0:
            logger.info(f"There are {len(completed_tasks)} awaiting evaluation")
            for task in completed_tasks:
                await _evaluate_task(task, config)
        if len(completed_tasks) == 0:
            logger.info('There are no tasks to evaluate - waiting 30 seconds')
            await asyncio.sleep(30)


async def process_pending_tasks(config: Config) -> None:
    while True:
        try:
            await _process_selected_tasks(config)
            await _find_miners_for_task(config)
            await _process_ready_to_train_tasks(config)
        except Exception as e:
            logger.info(f"There was a problem in processing: {e}")
            await asyncio.sleep(30)


async def validator_cycle(config: Config) -> None:
       try:
           await asyncio.gather(
               process_completed_tasks(config),
               process_pending_tasks(config),
           )
       except Exception as e:
           logger.error(f"Error in validator_cycle: {e}", exc_info=True)
           await asyncio.sleep(30)


async def node_refresh_cycle(config: Config) -> None:
    while True:
        try:
            logger.info("Attempting to refresh_nodes")
            await asyncio.wait_for(get_and_store_nodes(config), timeout=300)  # 5 minute timeout
            await asyncio.sleep(300)  # 5 minute wait between refreshes
        except asyncio.TimeoutError:
            logger.error("Node refresh timed out after 5 minutes")
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in node_refresh_cycle: {e}", exc_info=True)
            await asyncio.sleep(60)

# Not sure if this is the best solution to the problem of if something within the cycle crashes TT good with this stuff?
# If not, will come back - let me know  porfa
async def run_validator_cycles(config: Config) -> None:
    try:
        # Run node refresh cycle and validator cycle concurrently
        await asyncio.gather(
            node_refresh_cycle(config),
            _run_main_validator_loop(config)
        )
    except Exception as e:
        logger.error(f"Main validator cycles crashed: {e}", exc_info=True)
        await asyncio.sleep(30)

async def _run_main_validator_loop(config: Config) -> None:
    while True:
        try:
            node_list = await nodes_sql.get_all_nodes(config.psql_db)
            node_list = await perform_handshakes(node_list, config)
            logger.info(f"Current active nodes: {len(node_list)}")
            testdate = datetime.datetime.now() - datetime.timedelta(days=7)
            await scoring_aggregation_from_date(config.psql_db, testdate)
            await validator_cycle(config)
            # Add small sleep to prevent tight loop
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Validator loop iteration failed: {e}", exc_info=True)
            await asyncio.sleep(30)

def init_validator_cycles(config: Config) -> Task:
       return asyncio.create_task(run_validator_cycles(config))

