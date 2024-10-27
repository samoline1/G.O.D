from datetime import datetime
from typing import List
from typing import Optional
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field



class Task(BaseModel):
    task_id: Optional[UUID] = None
    model_id: str
    ds_id: str
    input: str
    status: str
    system: Optional[str] = None
    instruction: Optional[str] = None
    output: Optional[str] = None
    test_data: Optional[str] = None
    synthetic_data: Optional[str] = None
    hf_training_repo: Optional[str] = None
    assigned_miners: Optional[List[int]] = None
    miner_scores: Optional[List[float]] = None
    created_timestamp: Optional[datetime] = None
    updated_timestamp: Optional[datetime] = None
    started_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    hours_to_complete: int
    best_submission_repo: Optional[str] = None
    user_id: Optional[str] = None

class TaskNode(BaseModel):
    task_id: str
    node_id: int
    quality_score: float

class TaskResults(BaseModel):
    task: Task
    node_scores: list[TaskNode]


class NodeAggregationResult(BaseModel):
    work_sum: int = Field(default=0)
    work_sums: List[int] = Field(default_factory=list)
    average_score: Optional[float] = Field(default=0.0)
    summed_scores: float = Field(default=0.0)
    final_score: Optional[float] = Field(default=0.0)
    raw_scores: List[float] = Field(default_factory=list)
    node_id: int
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

class Node(BaseModel):
    node_id: int
    coldkey: str
    ip: str
    ip_type: str
    port: int
    symmetric_key: str
    network: float
    trust: Optional[float] = 0.0
    vtrust: Optional[float] = 0.0
    stake: float
    created_timestamp: Optional[datetime] = None
    updated_timestamp: Optional[datetime] = None


class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: Optional[float] = None
    task_id: UUID
    node_id: int
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]

class MinerResults(BaseModel):
    node_id: int
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: Optional[float] = 0.0
    submission: Optional[Submission] = None


