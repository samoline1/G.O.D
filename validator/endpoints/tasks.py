from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from fastapi import HTTPException
from loguru import logger

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.db import sql


# TODO request bodies should all be pydantic
async def create_task(
    model_id: str = Body(..., embed=True),
    ds_id: str = Body(..., embed=True),
    system: str = Body(..., embed=True),
    instruction: str = Body(..., embed=True),
    input_data: str = Body(..., embed=True),
    status: str = Body(..., embed=True),
    config: Config = Depends(get_config),
):
    task_id = await sql.add_task(model_id, ds_id, system, instruction, input_data, status, config.psql_db)
    logger.info(f"Task {task_id} created.")
    return {"success": True, "task_id": task_id}


async def get_task_status(
    task_id: str,
    config: Config = Depends(get_config),
):
    task = await sql.get_task(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    # TODO: reponses should be pydantic
    return {"success": True, "task_id": task_id, "status": task["status"]}


async def submit_task_submission(
    task_id: str = Body(..., embed=True),
    node_id: str = Body(..., embed=True),
    repo: str = Body(..., embed=True),
    config: Config = Depends(get_config),
):

    # TODO: This needs implementing in the db side
    is_unique = await sql.submission_repo_is_unique(repo, config.psql_db)
    is_miner_assigned_to_task = await sql.is_miner_assigned_to_task(task_id, node_id, config.psql_db)
    if not is_unique:
        return {"success": False, "message": "Submission with this repository already exists."}
    elif not is_miner_assigned_to_task:
        return {"success": False, "message": "You are not registered as assigned to this task."}
    else:
        submission_id = await sql.add_submission(task_id, node_id, repo, config.psql_db)
        return {"success": True, "message": "sucess"}



def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/tasks/create",
        create_task,
        tags=["tasks"],
        methods=["POST"],
    )

    router.add_api_route(
        "/tasks/status/{task_id}",
        get_task_status,
        tags=["tasks"],
        methods=["GET"],
    )

    router.add_api_route(
        "/tasks/submit",
        submit_task_submission,
        tags=["tasks"],
        methods=["POST"],
    )

    return router
