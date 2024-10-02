from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, Optional, Literal

class DatasetType(str, Enum):
    INSTRUCT = "instruct"
    PRETRAIN = "pretrain"
    ALPACA = "alpaca"

class FileFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    HF = "hf"

class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"

class CustomDatasetType(BaseModel):
    system_prompt: Optional[str]
    system_format: Optional[str]
    field_system: Optional[str]
    field_instruction: Optional[str]
    field_input: Optional[str]
    field_output: Optional[str]
    format: Optional[str]
    no_input_format: Optional[str]
    field: Optional[str]

class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Path to the dataset file or Hugging Face dataset name")
    model: str = Field(..., description="Name or path of the model to be trained")
    dataset_type: Union[DatasetType, CustomDatasetType]
    file_format: FileFormat

class TrainResponse(BaseModel):
    message: str
    job_id: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
