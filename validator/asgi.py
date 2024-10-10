import asyncio
import os

import httpx
from dotenv import load_dotenv

load_dotenv(os.getenv("ENV_FILE", ".env"))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from validator.core.config import factory_config
from validator.core.cycle import init_validator_cycle

from validator.db import sql

from validator.endpoints.health import factory_router as health_router




@asynccontextmanager
async def lifespan(app: FastAPI):
    config = factory_config()
    await config.psql_db.connect()

    # Add configuration to app state for dependency access
    app.state.config = config

    init_validator_cycle(config)

    yield

    logger.info("Shutting down...")
    await config.psql_db.close()
    await config.redis_db.aclose()


def factory() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    
    # Include routers
    app.include_router(health_router())

    # Add CORS middleware
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
