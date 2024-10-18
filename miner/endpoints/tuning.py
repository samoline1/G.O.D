from fiber.logging_utils import get_logger
import json
import os
import yaml
from urllib.parse import urlparse

import aiohttp
from fastapi import Depends
from fastapi import HTTPException
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger

from core.models.payload_models import MinerTaskRequst
from core.models.payload_models import MinerTaskResponse
from core.models.payload_models import TrainRequest
from core.models.payload_models import TrainResponse
from core.models.utility_models import FileFormat
from core.utils import validate_dataset
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job
from validator.core.models import Node
from validator.utils.call_endpoint import process_non_stream
from validator.utils.minio import async_minio_client
import core.constants as cst


logger = get_logger(__name__)

async def download_s3_file(file_url: str) -> str:
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    local_file_path = os.path.join("/tmp", file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, 'wb') as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path

async def tune_model(
    decrypted_payload: TrainRequest,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    logger.info("Starting model tuning.")
    logger.info(f"Job received is {decrypted_payload}")

    if not decrypted_payload.dataset or not decrypted_payload.model:
        raise HTTPException(status_code=400, detail="Dataset and model are required.")

    try:
        logger.info(decrypted_payload.file_format)
        if decrypted_payload.file_format != FileFormat.HF:
            if decrypted_payload.file_format == FileFormat.S3:
                decrypted_payload.dataset = await download_s3_file(decrypted_payload.dataset)
                logger.info(decrypted_payload.dataset)
                decrypted_payload.file_format = FileFormat.JSON

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        job_id= str(decrypted_payload.task_id),
        dataset=decrypted_payload.dataset,
        model=decrypted_payload.model,
        dataset_type=decrypted_payload.dataset_type,
        file_format=decrypted_payload.file_format,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}

async def get_latest_model_submission(task_id: str) -> str:
    try:
        config_filename = f"{task_id}.yml"
        config_path = os.path.join(cst.CONFIG_STORAGE_DIR, config_filename)
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            return config_data.get('hub_model_id', None)

    except Exception as e:
        logger.error(f"Error retrieving latest model submission for task {task_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"No model submission found for task {task_id}")


async def task_offer(request: MinerTaskRequst) -> MinerTaskResponse:
    if request.hours_to_complete < 100:
        return MinerTaskResponse(message='Yes', accepted=True)
    else:
        return MinerTaskResponse(message='I only accept small jobs', accepted=False)


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/task_offer/",
        task_offer,
        tags=["Subnet"],
        methods=["POST"],
        response_model=MinerTaskResponse
    )
    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID"
    )
    router.add_api_route(
        "/start_training/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
    )
    return router
