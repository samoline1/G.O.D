import os

from dotenv import load_dotenv


load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from validator.core.config import factory_config
from validator.core.cycle import init_validator_cycle
from validator.endpoints.health import factory_router as health_router
from validator.endpoints.nodes import factory_router as nodes_router
from validator.endpoints.tasks import factory_router as tasks_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = factory_config()
    await config.psql_db.connect()

    app.state.config = config

    init_validator_cycle(config)

    yield

    logger.info("Shutting down...")
    await config.psql_db.close()
    await config.redis_db.aclose()


def factory() -> FastAPI:
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

    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(factory(), host="0.0.0.0", port=8010)
