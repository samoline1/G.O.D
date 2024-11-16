from datetime import datetime
from typing import List
from typing import Optional
from uuid import UUID
from uuid import uuid4
from cryptography.fernet import Fernet

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
    delay_timestamp: Optional[datetime] = None
    updated_timestamp: Optional[datetime] = None
    started_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    hours_to_complete: int
    best_submission_repo: Optional[str] = None
    user_id: Optional[str] = None


class PeriodScore(BaseModel):
    quality_score: float
    summed_task_score: float
    average_score: float
    hotkey: str
    normalised_score: Optional[float] = 0.0

class TaskNode(BaseModel):
    task_id: str
    hotkey: str
    quality_score: float

class TaskResults(BaseModel):
    task: Task
    node_scores: list[TaskNode]

class NodeAggregationResult(BaseModel):
    task_work_scores: List[float] = Field(default_factory=list)
    average_raw_score: Optional[float] = Field(default=0.0)
    summed_adjusted_task_scores: float = Field(default=0.0)
    quality_score: Optional[float] = Field(default=0.0)
    emission: Optional[float] = Field(default=0.0)
    task_raw_scores: List[float] = Field(default_factory=list)
    hotkey: str
    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

class Submission(BaseModel):
    submission_id: UUID = Field(default_factory=uuid4)
    score: Optional[float] = None
    task_id: UUID
    hotkey: str
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]

class MinerResults(BaseModel):
    hotkey: str
    test_loss: float
    synth_loss: float
    is_finetune: bool
    score: Optional[float] = 0.0
    submission: Optional[Submission] = None

class LeaderboardRow(BaseModel):
    hotkey: str
    average_quality_score: float
    sum_quality_score: float
    num_games_entered: int
