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
    system_prompt: Optional[str] = None
    system_format: Optional[str] = "{system}"
    field_system: Optional[str] = None
    field_instruction: Optional[str] = None
    field_input: Optional[str] = None
    field_output: Optional[str] = None
    format: Optional[str] = None
    no_input_format: Optional[str] = None
    field: Optional[str] = None

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
