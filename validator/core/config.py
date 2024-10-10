import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from fiber import SubstrateInterface
from fiber.chain import interface
from redis.asyncio import Redis

from validator.db.database import PSQLDB


load_dotenv()


@dataclass
class Config:
    psql_db: PSQLDB
    redis_db: Redis
    # substrate: SubstrateInterface
    httpx_client: httpx.AsyncClient = httpx.AsyncClient(timeout=5)


def factory_config() -> Config:
    # subtensor_network = os.getenv("SUBTENSOR_NETWORK")
    # subtensor_address = os.getenv("SUBTENSOR_ADDRESS")
    # substrate = interface.get_substrate(
    #     subtensor_network=subtensor_network,
    #     subtensor_address=subtensor_address,
    # )

    localhost = bool(os.getenv("LOCALHOST", "false").lower() == "true")

    if localhost:
        redis_host = "localhost"
        os.environ["POSTGRES_HOST"] = "localhost"
    else:
        redis_host = os.getenv("REDIS_HOST", "redis")

    redis_url = os.getenv("REDIS_URL")
    if redis_url is None:
        redis = Redis(host=redis_host)
    else:
        redis = Redis.from_url(redis_url)

    return Config(
        psql_db=PSQLDB(),
        redis_db=redis,
        # substrate=substrate,
    )
