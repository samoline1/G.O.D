import os

import uvicorn
from dotenv import load_dotenv


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fiber.logging_utils import get_logger

from validator.core.config import factory_config
from validator.core.cycle import init_validator_cycle
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.nodes import factory_router as nodes_router
from validator.endpoints.tasks import factory_router as tasks_router


logger = get_logger(__name__)

import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug("Entering lifespan context manager")
    config = factory_config()
    logger.debug(f"Config created: {config}")

    try:
        logger.debug("Attempting to connect to PostgreSQL")
        await asyncio.wait_for(config.psql_db.connect(), timeout=5.0)
        logger.debug("PostgreSQL connected successfully")
    except asyncio.TimeoutError:
        logger.error("Timeout while connecting to PostgreSQL")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")

    try:
        logger.debug("Attempting to connect to Redis")
        await asyncio.wait_for(config.redis_db.ping(), timeout=5.0)
        logger.debug("Redis connected successfully")
    except asyncio.TimeoutError:
        logger.error("Timeout while connecting to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")

    logger.info('Starting up...')
    app.state.config = config

    try:
        logger.debug("Initializing validator cycle")
        init_validator_cycle(config)
        logger.debug("Validator cycle initialized")
    except Exception as e:
        logger.error(f"Failed to initialize validator cycle: {e}")

    yield

    logger.info("Shutting down...")
    await config.psql_db.close()
    await config.redis_db.close()

def factory() -> FastAPI:
    logger.debug("Entering factory function")
    app = FastAPI(lifespan=lifespan)

    app.include_router(health_router())
    app.include_router(tasks_router())
    app.include_router(nodes_router())

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.debug(f"App created with {len(app.routes)} routes")
    return app

if __name__ == "__main__":

    logger.info('Starting main validator')

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
