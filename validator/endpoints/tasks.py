from datetime import datetime
from datetime import timedelta
from typing import List
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Response
from fastapi import Security
from fastapi.security import APIKeyHeader
from fiber.logging_utils import get_logger

from core.models.payload_models import NewTaskRequest
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import TaskStatusResponse
from core.models.utility_models import TaskStatus
from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.core.models import Task
from validator.db import sql


logger = get_logger(__name__)

# NOTE: please dont just dump AI code in here without thinking - this is how bugs get in
# Define a custom security scheme for the Bearer token
bearer_token_header = APIKeyHeader(name="Authorization", auto_error=False)


async def delete_task(
    task_id: UUID,
    # What is the comment below for - its obvious right?
    authorization: str = Security(bearer_token_header),  # Use Security with APIKeyHeader
    # Instead of the above you need to make a dependency and
    # extract the user_id from the token, and use that here
    # This is a byproduct of pasting gpt code
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token is required.")

    user_id = authorization

    task = await sql.get_task(task_id, config.psql_db)

    # If task is none you mean?
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this task.")

    await sql.delete_task(task_id, config.psql_db)
    return Response(success=True)


async def get_tasks(
    authorization: str = Security(bearer_token_header),  # Use Security with APIKeyHeader
    config: Config = Depends(get_config),
) -> List[TaskStatusResponse]:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token is required.")

    user_id = authorization

    tasks_with_miners = await sql.get_tasks_with_miners_by_user(user_id, config.psql_db)

    logger.info(tasks_with_miners)

    # Why so many magic strings? can't you keep it as a Task object?

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
    # gpt code be gone plz
    authorization: str = Security(bearer_token_header),  # Use Security with APIKeyHeader
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization token is required.")

    user_id = authorization
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
        user_id=user_id,
    )

    logger.info(f"The Task is {task}")

    task = await sql.add_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id)


async def get_task_status(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> TaskStatusResponse:
    task = await sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    return TaskStatusResponse(
        success=True,
        task_id=task_id,
        status=task.status,
        model_id=task.model_id,
        miners=None,
        dataset=task.hf_training_repo,
        created=str(task.created_timestamp),  # created_at? is str() ok here?
        hours_to_complete=task.hours_to_complete,
    )



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
        "/tasks/{task_id}",
        get_task_status,
        response_model=TaskStatusResponse,
        tags=["tasks"],
        methods=["GET"],
    )

    router.add_api_route(
        "/tasks/delete/{task_id}",
        delete_task,
        response_model=NewTaskResponse,
        tags=["tasks"],
        methods=["DELETE"],
    )

    router.add_api_route(
        "/tasks/",
        get_tasks,
        response_model=List[TaskStatusResponse],
        tags=["tasks"],
        methods=["GET"],
    )

    return router
