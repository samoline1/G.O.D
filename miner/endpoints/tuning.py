import os
from urllib.parse import urlparse
import aiohttp
import yaml
from fastapi import Depends
from fastapi import Request
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger

import core.constants as cst
from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job

from fastapi import Depends, APIRouter
from functools import partial
from fiber.miner.security.encryption import decrypt_general_payload
from fiber.miner.dependencies import blacklist_low_stake, verify_request
logger = get_logger(__name__)


async def tune_model(
    request: TrainRequest,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info("Starting model tuning.")
    logger.info(f"Job received is {request}")

    if not request.dataset or not request.model:
        raise HTTPException(status_code=400, detail="Dataset and model are required.")

    try:
        logger.info(request.file_format)
        if request.file_format != FileFormat.HF:
            if request.file_format == FileFormat.S3:
                request.dataset = await download_s3_file(request.dataset)
                logger.info(request.dataset)
                request.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        job_id=str(request.task_id),
        dataset=request.dataset,
        model=request.model,
        dataset_type=request.dataset_type,
        file_format=request.file_format,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


async def get_latest_model_submission(task_id: str) -> str:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_DIR, config_filename)
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
            return config_data.get("hub_model_id", None)

    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")


async def task_offer(
    decrypted_payload: MinerTaskRequst = Depends(partial(decrypt_general_payload, MinerTaskRequst)),
) -> MinerTaskResponse:
    try:
        logger.info(f"Got a descryted payload {decrypted_payload}")
        if decrypted_payload.hours_to_complete < 100:
            return MinerTaskResponse(message="Yes", accepted=True)
        else:
            return MinerTaskResponse(message="I only accept small jobs", accepted=False)
    except Exception as e:
        logger.error(f"Error processing task offer: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")

def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/task_offer/", task_offer, tags=["Subnet"], methods=["POST"], response_model=MinerTaskResponse)
    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
    )
    router.add_api_route(
        "/start_training/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
    )
    return router
