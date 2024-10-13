import asyncio

from loguru import logger

from validator.db import sql


async def validator_cycle(config):
    try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            tasks = await sql.get_tasks_by_status('training', config.psql_db)
            logger.info(tasks)
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise


def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))
