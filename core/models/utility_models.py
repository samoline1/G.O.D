from pydantic import BaseModel
from enum import Enum
import uuid
from pydantic import Field


class DatasetType(str, Enum):
    INSTRUCT = "instruct"
    PRETRAIN = "pretrain"
    ALPACA = "alpaca"
    # there is actually loads of these supported, but laziness is key here, add when we need


class FileFormat(str, Enum):
    CSV = "csv"  # needs to be local file
    JSON = "json"  # needs to be local file
    HF = "hf"  # Hugging Face dataset


class JobStatus(str, Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    NOT_FOUND = "Not Found"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IDLE = "idle"
    SUCCESS = "success"
    EVALUATING = "evaluating"
    TRAINING = "training"
    FAILURE = "failure"

class CustomDatasetType(BaseModel):
    system_prompt: str | None = None
    system_format: str | None = "{system}"
    field_system: str | None = None
    field_instruction: str | None = None
    field_input: str | None = None
    field_output: str | None = None
    format: str | None = None
    no_input_format: str | None = None
    field: str | None = None

class Job(BaseModel):
    task_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    dataset: str
    model: str
    dataset_type: DatasetType | CustomDatasetType
    file_format: FileFormat
    status: JobStatus = JobStatus.QUEUED
    error_message: str | None = None


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role
    content: str

class Prompts(BaseModel):
    synth_data_creation_sys: str
    synth_data_creation_prompt: str
