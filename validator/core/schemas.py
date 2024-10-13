from datetime import datetime
from typing import List
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Task(BaseModel):
    task_id: Optional[UUID]
    model_id: str
    ds_id: str
    system: str
    instruction: str
    input: str
    output: Optional[str]
    status: str
    test_data: Optional[str]
    synthetic_data: Optional[str]
    hf_training_repo: Optional[str]
    miner_scores: Optional[List[float]]
    created_timestamp: Optional[datetime]
    updated_timestamp: Optional[datetime]
    started_timestamp: Optional[datetime]
    completed_timestamp: Optional[datetime]
