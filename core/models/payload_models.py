from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from uuid import UUID

from fiber.logging_utils import get_logger
from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import JobStatus
from core.models.utility_models import TaskStatus


logger = get_logger(__name__)

class MinerTaskRequst(BaseModel):
    ds_size: int
    model: str
    hours_to_complete: int
    task_id: str

class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Path to the dataset file or Hugging Face dataset name")
    model: str = Field(..., description="Name or path of the model to be trained")
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    task_id: str
    hours_to_complete: int


class TrainResponse(BaseModel):
    message: str
    task_id: UUID


class JobStatusPayload(BaseModel):
    task_id: UUID


class JobStatusResponse(BaseModel):
    task_id: UUID
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str


class EvaluationResult(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class TaskRequest(BaseModel):
    ds_repo: str
    system_col: str
    instruction_col: str
    input_col: str
    output_col: str
    model_repo: str
    hours_to_complete: int


class SubmitTaskSubmissionRequest(BaseModel):
    task_id: str
    node_id: int
    repo: str


class TaskResponse(BaseModel):
    success: bool
    task_id: str
    message: Optional[str] = None


class SubmissionResponse(BaseModel):
    success: bool
    message: str
    submission_id: Optional[str] = None


class NewTaskRequest(BaseModel):
    model_repo: str
    ds_repo: str
    input_col: str
    hours_to_complete: int
    system_col: Optional[str] = None
    output_col: Optional[str] = None
    instruction_col: Optional[str] = None

class GetTasksRequest(BaseModel):
    fingerprint: str

class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID

class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str


class MinerTaskResponse(BaseModel):
    hotkey: str
    quality_score: float

class TaskResultResponse(BaseModel):
    success: bool
    id: UUID
    miner_results: Optional[list[MinerTaskResponse]]



class TaskStatusResponse(BaseModel):
    success: bool
    id: UUID
    status: TaskStatus
    miners: Optional[List[Dict]]
    model_repo: str
    ds_repo: Optional[str]
    input_col: Optional[str]
    system_col: Optional[str]
    output_col: Optional[str]
    instruction_col: Optional[str]
    started: str
    end: str
    created: str
    hours_to_complete: int
    winning_submission: Optional[Union[WinningSubmission, None]] = None

class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus
