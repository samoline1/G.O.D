from datetime import datetime
from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fiber.logging_utils import get_logger

from core.models.payload_models import NewTaskRequest
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import TaskStatusResponse
from core.models.payload_models import TaskSubmissionRequest
from core.models.payload_models import TaskSubmissionResponse
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.db import sql

logger = get_logger(__name__)

async def create_task(
    request: NewTaskRequest,
    config: Config = Depends(get_config),
) -> NewTaskResponse:

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = Task(
        model_id=request.model_repo,
        ds_id=request.ds_repo,
        system=request.system_col,
        instruction=request.instruction_col,
        input=request.input_col,
        output=request.output_col,
        status=TaskStatus.PENDING,
        end_timestamp=end_timestamp,
        hours_to_complete=request.hours_to_complete
    )

    task = await sql.add_task(
        task,
        config.psql_db
    )

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id)

async def get_task_status(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> TaskStatusResponse:
    task = await sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(success=True, task_id=task_id, status=task.status)

async def submit_task_submission(
    request: TaskSubmissionRequest,
    config: Config = Depends(get_config),
) -> TaskSubmissionResponse:
    is_unique = await sql.submission_repo_is_unique(request.repo, config.psql_db)
    is_miner_assigned_to_task = await sql.is_miner_assigned_to_task(request.task_id, request.node_id, config.psql_db)

    if not is_unique:
        return TaskSubmissionResponse(success=False, message="Submission with this repository already exists.")
    elif not is_miner_assigned_to_task:
        return TaskSubmissionResponse(success=False, message="You are not registered as assigned to this task.")
    else:
        submission_id = await sql.add_submission(request.task_id, request.node_id, request.repo, config.psql_db)
        return TaskSubmissionResponse(success=True, message="success", submission_id=submission_id)

def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/tasks/create",
        create_task,
        response_model=NewTaskResponse,
        tags=["tasks"],
        methods=["POST"],
    )

    router.add_api_route(
        "/tasks/status/{task_id}",
        get_task_status,
        response_model=TaskStatusResponse,
        tags=["tasks"],
        methods=["GET"],
    )

    router.add_api_route(
        "/tasks/submit",
        submit_task_submission,
        response_model=TaskSubmissionResponse,
        tags=["tasks"],
        methods=["POST"],
    )

    return router
