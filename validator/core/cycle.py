import asyncio
from loguru import logger


async def validator_cycle(config):
    """
    A simple validator cycle that prints "Hello, World!" every 5 seconds.
    """
    try:
        while True:
            logger.info("Validator Heartbeat! Its alive!")
            await asyncio.sleep(5)
    except asyncio.CancelledError:
        logger.info("Validator cycle cancelled, shutting down...")
        raise


def init_validator_cycle(config):
    return asyncio.create_task(validator_cycle(config))