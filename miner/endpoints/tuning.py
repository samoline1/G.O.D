import os
from urllib.parse import urlparse
import aiohttp
import yaml
from fastapi import Depends
from fastapi import Request
from fastapi import HTTPException
from fastapi.routing import APIRouter

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
from fiber.miner.dependencies import blacklist_low_stake, get_config, verify_request
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from datetime import datetime, timedelta

logger = get_logger(__name__)

finish_time = None


async def tune_model(
    decrypted_payload: TrainRequest = Depends(partial(decrypt_general_payload, TrainRequest)),
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    global finish_time
    logger.info("Starting model tuning.")

    finish_time = datetime.now() + timedelta(hours=decrypted_payload.hours_to_complete)
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
        job_id=str(decrypted_payload.task_id),
        dataset=decrypted_payload.dataset,
        model=decrypted_payload.model,
        dataset_type=decrypted_payload.dataset_type,
        file_format=decrypted_payload.file_format,
    )
    logger.info(f"Created job {job}")
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "task_id": job.job_id}


# I think we need to be v careful that it's validators that are asking for this, is there a way to ensure we only reply to validators?
async def get_latest_model_submission(
    task_id: str,
) -> str:
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
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
) -> MinerTaskResponse:
    try:
        global finish_time
        current_time = datetime.now()

        if finish_time is None or current_time + timedelta(hours=1) > finish_time:
            if decrypted_payload.hours_to_complete < 100:
                return MinerTaskResponse(message="Yes", accepted=True)
            else:
                return MinerTaskResponse(message="I only accept small jobs", accepted=False)
        else:
            return MinerTaskResponse(
                message=f"Currently busy with another job until {finish_time.isoformat()}",
                accepted=False
            )

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in task_offer: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing task offer: {str(e)}")

def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route("/task_offer/",
                         task_offer,
                         tags=["Subnet"], methods=["POST"],
                         response_model=MinerTaskResponse,
                         dependencies=[Depends(blacklist_low_stake), Depends(verify_request)])
    router.add_api_route(
        "/get_latest_model_submission/{task_id}",
        get_latest_model_submission,
        tags=["Subnet"],
        methods=["GET"],
        response_model=str,
        summary="Get Latest Model Submission",
        description="Retrieve the latest model submission for a given task ID",
        dependencies=[Depends(blacklist_low_stake)]
    )
    router.add_api_route(
        "/start_training/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        response_model=TrainResponse,
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)]
    )
    return router
