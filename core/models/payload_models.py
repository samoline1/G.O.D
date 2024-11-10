from typing import List
from typing import Optional
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
    fingerprint: str
    system_col: Optional[str] = None
    output_col: Optional[str] = None
    instruction_col: Optional[str] = None

class GetTasksRequest(BaseModel):
    fingerprint: str

class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID


class TaskStatusResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus
    miners: Optional[List]
    model_id: str
    dataset: Optional[str]
    created: str
    hours_to_complete: int
