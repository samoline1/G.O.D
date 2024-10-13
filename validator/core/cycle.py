import asyncio

from loguru import logger

from validator.db import sql
from validator.tasks.task_prep import prepare_task
from core.models.utility_models import TaskStatus

def run_task_prep(task: Task):
    output_task_repo_name = task.id
    test_data, synth_data = prepare_task(dataset_name=task.repo_name, columns_to_sample=[task.system, task.instruction, task.input_data, task.output], repo_name=output_task_repo_name)
    task.status =  TaskStatus.TRAINING
    await update_task_info_in_the_db(task)
    return task

async def validator_cycle(config):
     try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            tasks = await sql.get_tasks_by_status('pending', config.psql_db) # assuming we have a task object coming back
            for task in tasks:
                task = await run_task_prep(task)
                pass
            logger.info(tasks)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
