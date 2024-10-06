from typing import Any
from pydantic import BaseModel, Field

from core.models.utility_models import CustomDatasetType, DatasetType, FileFormat, JobStatus


class CapacityResponse(BaseModel):
    capacities: dict[str, float]


class CapacityPayload(BaseModel):
    task_configs: list[dict[str, Any]]


class TuningPayload(BaseModel): ...


class TrainRequest(BaseModel):
    dataset: str = Field(
        ..., description="Path to the dataset file or Hugging Face dataset name"
    )
    model: str = Field(..., description="Name or path of the model to be trained")
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat


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


class EvaluationResponse(BaseModel):
    is_finetune: bool
    eval_results: dict[str, Any]
