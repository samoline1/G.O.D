from typing import Optional
from uuid import UUID

from pydantic import BaseModel
from pydantic import Field

from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import JobStatus
from core.models.utility_models import TaskStatus


class TrainRequest(BaseModel):
    dataset: UUID = Field(
        ..., description="Path to the dataset file or Hugging Face dataset name"
    )
    model: str = Field(..., description="Name or path of the model to be trained")
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    job_id: str

class TrainResponse(BaseModel):
    message: str
    job_id: str


class JobStatusPayload(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str

class EvaluationResult(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float

class MinerTaskRequst(BaseModel):
    hf_training_repo: str
    model: str
    hours_to_complete: int

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
    node_id: str
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
    system_col: str
    instruction_col: str
    input_col: Optional[str] = None
    output_col: Optional[str] = None
    hours_to_complete: Optional[float] = None


class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID

class TaskStatusResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus

class TaskSubmissionRequest(BaseModel):
    task_id: UUID
    node_id: UUID
    repo: str

class TaskSubmissionResponse(BaseModel):
    success: bool
    message: str
    submission_id: Optional[UUID] = None
