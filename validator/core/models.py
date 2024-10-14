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
    hours_to_complete: int
    created_timestamp: Optional[datetime]
    updated_timestamp: Optional[datetime]
    started_timestamp: Optional[datetime]
    completed_timestamp: Optional[datetime]

class Node(BaseModel):
    node_id: Optional[UUID]
    coldkey: str
    ip: str
    ip_type: str
    port: int
    symmetric_key: str
    network: float
    trust: Optional[float]
    vtrust: Optional[float]
    stake: float
    created_timestamp: Optional[datetime]
    updated_timestamp: Optional[datetime]


class Submission(BaseModel):
    submission_id: Optional[UUID]
    task_id: UUID
    node_id: UUID
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]

