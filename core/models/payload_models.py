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
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    task_id: str
    hours_to_complete: int


class TrainRequestText(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat


class TrainRequestDiffusion(TrainRequest):
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )


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


class EvaluationRequestDiffusion(BaseModel):
    test_split_path: str
    base_model_repo: str
    base_model_filename: str
    lora_repo_list: str
    lora_filename_list: str


class EvaluationResult(BaseModel):
    is_finetune: bool
    eval_loss: float
    perplexity: float


class EvaluationResultDiffusion(BaseModel):
    eval_losses: Dict


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class TaskRequest(BaseModel):  # did not add format
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
    instruction_col: str
    input_col: Optional[str] = None
    hours_to_complete: int
    system_col: Optional[str] = None
    output_col: Optional[str] = None
    format_col: Optional[str] = None
    no_input_format_col: Optional[str] = None


class GetTasksRequest(BaseModel):
    fingerprint: str


class NewTaskResponse(BaseModel):
    success: bool
    task_id: UUID


class WinningSubmission(BaseModel):
    hotkey: str
    score: float
    model_repo: str


class MinerTaskResult(BaseModel):
    hotkey: str
    quality_score: float


class TaskMinerResult(BaseModel):
    task_id: UUID
    quality_score: float


class AllOfNodeResults(BaseModel):
    success: bool
    hotkey: str
    task_results: Optional[list[TaskMinerResult]]


class TaskResultResponse(BaseModel):
    success: bool
    id: UUID
    miner_results: Optional[list[MinerTaskResult]]


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
    format_col: Optional[str]
    no_input_format_col: Optional[str]
    started: str
    end: str
    created: str
    hours_to_complete: int
    winning_submission: Optional[Union[WinningSubmission, None]] = None


class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus
