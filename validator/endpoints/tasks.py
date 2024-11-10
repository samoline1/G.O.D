from datetime import datetime
from datetime import timedelta
from typing import List
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fiber.logging_utils import get_logger

from core.models.payload_models import NewTaskRequest
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import TaskStatusResponse
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.db.sql import tasks as task_sql


logger = get_logger(__name__)

async def delete_task(
    task_id: UUID,
    user_id: str = Depends(get_api_key),
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> NewTaskResponse:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this task.")

    await task_sql.delete_task(task_id, config.psql_db)
    return Response(success=True)


async def get_tasks(
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> List[TaskStatusResponse]:
    tasks_with_miners = await task_sql.get_tasks_with_miners_by_user(request.fingerprint, config.psql_db)

    logger.info(tasks_with_miners)

    return [
        TaskStatusResponse(
            success=True,
            task_id=task["task_id"],
            status=task["status"],
            miners=task["miners"],
            model_id=task["model_id"],
            dataset=task["hf_training_repo"],
            created=task["created_timestamp"].strftime("%Y-%m-%dT%H:%M:%S")
            if isinstance(task["created_timestamp"], datetime)
            else task["created_timestamp"],
            hours_to_complete=task["hours_to_complete"],
        )
        for task in tasks_with_miners
    ]


async def create_task(
    request: NewTaskRequest,
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    logger.info(f"The request coming in {request}")
    task = Task(
        model_id=request.model_repo,
        ds_id=request.ds_repo,
        system=request.system_col,
        instruction=request.instruction_col,
        input=request.input_col,
        output=request.output_col,
        status=TaskStatus.PENDING,
        end_timestamp=end_timestamp,
        hours_to_complete=request.hours_to_complete,
    )

    logger.info(f"The Task is {task}")

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id)


async def get_task_status(
    task_id: UUID,
    config: Config = Depends(get_config),
    api_key: str = Depends(get_api_key),
) -> TaskStatusResponse:
    task = await task_sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    return TaskStatusResponse(
        success=True,
        task_id=task_id,
        status=task.status,
        model_id=task.model_id,
        miners=None,
        dataset=task.hf_training_repo,
        created=str(task.created_timestamp),
        hours_to_complete=task.hours_to_complete,
    )


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/v1/tasks/create",
        create_task,
        response_model=NewTaskResponse,
        tags=["Training"],
        methods=["POST"],
    )

    router.add_api_route(
        "/v1/tasks/{task_id}",
        get_task_status,
        response_model=TaskStatusResponse,
        tags=["Training"],
        methods=["GET"],
    )

    router.add_api_route(
        "/v1/tasks/delete/{task_id}",
        delete_task,
        response_model=NewTaskResponse,
        tags=["Training"],
        methods=["DELETE"],
    )

    router.add_api_route(
        "/v1/tasks/",
        get_tasks,
        response_model=List[TaskStatusResponse],
        tags=["Training"],
        methods=["GET"],
    )

    return router
