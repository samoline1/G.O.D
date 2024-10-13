import asyncio

from loguru import logger

from validator.db import sql
from validator.tasks.task_prep import prepare_task
from validator.core.models import Task
from core.models.utility_models import TaskStatus

def run_task_prep(task: Task) -> Task:
    output_task_repo_name = task.id
    test_data, synth_data = prepare_task(dataset_name=task.repo_name, columns_to_sample=[task.system, task.instruction, task.input_data, task.output], repo_name=output_task_repo_name)
    task.status =  TaskStatus.TRAINING
    return task

async def validator_cycle(config):
     try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            tasks = await sql.get_tasks_by_status(TaskStatus.PENDING, config.psql_db)
            for task in tasks:
                task = run_task_prep(task)
                await sql.update_task(task, config.psql_db)
            completed_tasks = await sql.get_tasks_ready_for_evaluation(config.psql_db)
            for completed_task in completed_tasks:
                evaluate_and_score(completed_task, config)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise

def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
