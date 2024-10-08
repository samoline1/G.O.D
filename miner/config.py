from dataclasses import dataclass
from functools import lru_cache


from dotenv import load_dotenv
from typing import TypeVar
from pydantic import BaseModel

from miner.logic.training_worker import TrainingWorker

load_dotenv()


T = TypeVar("T", bound=BaseModel)


@dataclass
class WorkerConfig:
    trainer: TrainingWorker


@lru_cache
def factory_worker_config() -> WorkerConfig:
    return WorkerConfig(
        trainer=TrainingWorker(),
    )
