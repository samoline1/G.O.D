from datetime import datetime
from typing import List
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class Task(BaseModel):
    task_id: Optional[UUID] = None
    model_id: str
    ds_id: str
    system: str
    instruction: str
    input: str
    output: Optional[str] = None
    status: str
    test_data: Optional[str] = None
    synthetic_data: Optional[str] = None
    hf_training_repo: Optional[str] = None
    miner_scores: Optional[List[float]] = None
    created_timestamp: Optional[datetime] = None
    updated_timestamp: Optional[datetime] = None
    started_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    completed_timestamp: Optional[datetime] = None
    hours_to_complete: int

    # task = Task(
    #     model_id=request.model_repo,
    #     ds_id=request.ds_repo,
    #     system=request.system_col,
    #     instruction=request.instruction_col,
    #     input=request.input_col,
    #     output=request.output_col,
    #     status=TaskStatus.PENDING,
    #     end_timestamp=end_timestamp
    # )

# task_id
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# test_data
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# synthetic_data
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# hf_training_repo
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# miner_scores
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# created_timestamp
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# updated_timestamp
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# started_timestamp
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing
# completed_timestamp
#   Field required [type=missing, input_value={'model_id': 'string', 'd...14, 15, 16, 51, 875476)}, input_type=dict]
#     For further information visit https://errors.pydantic.dev/2.9/v/missing

class Node(BaseModel):
    node_id: Optional[UUID]
    coldkey: str
    ip: str
    ip_type: str
    port: int
    symmetric_key: str
    network: float
    trust: Optional[float]
    vtrust: Optional[float]
    stake: float
    created_timestamp: Optional[datetime]
    updated_timestamp: Optional[datetime]


class Submission(BaseModel):
    submission_id: Optional[UUID]
    task_id: UUID
    node_id: UUID
    repo: str
    created_on: Optional[datetime]
    updated_on: Optional[datetime]
