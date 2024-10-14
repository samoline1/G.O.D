from core.models.utility_models import TaskStatus
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from loguru import logger
from core.models import Task
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.db import sql
from core.models.payload_models import TaskRequest, TaskResponse, TaskStatusResponse, SubmitTaskSubmissionRequest, SubmissionResponse

async def create_task(
    request: TaskRequest,
    config: Config = Depends(get_config),
) -> TaskResponse:

    task = Task(model_id=request.model_id,
         ds_id = request.ds_id,
         system =request.sytem,
         instruction= request.instruction,
         input= request.input,
         hours_to_complete= request.hours_to_complete,
         status=TaskStatus.PENDING
         )
    task_id = await sql.add_task(
        task,
        config.psql_db
    )
    logger.info(f"Task {task_id} created.")
    return TaskResponse(success=True, task_id=task_id)

async def get_task_status(
    task_id: str,
    config: Config = Depends(get_config),
) -> TaskStatusResponse:
    task = await sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskStatusResponse(success=True, task_id=task_id, status=task["status"])

async def submit_task_submission(
    request: SubmitTaskSubmissionRequest,
    config: Config = Depends(get_config),
) -> SubmissionResponse:
    is_unique = await sql.submission_repo_is_unique(request.repo, config.psql_db)
    is_miner_assigned_to_task = await sql.is_miner_assigned_to_task(request.task_id, request.node_id, config.psql_db)

    if not is_unique:
        return SubmissionResponse(success=False, message="Submission with this repository already exists.")
    elif not is_miner_assigned_to_task:
        return SubmissionResponse(success=False, message="You are not registered as assigned to this task.")
    else:
        submission_id = await sql.add_submission(request.task_id, request.node_id, request.repo, config.psql_db)
        return SubmissionResponse(success=True, message="success", submission_id=submission_id)

def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/tasks/create",
        create_task,
        response_model=TaskResponse,
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
        response_model=SubmissionResponse,
        tags=["tasks"],
        methods=["POST"],
    )

    return router
