import json
import os

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


logger = get_logger(__name__)

async def download_s3_file(dataset_path: str) -> str:

    logger.info(dataset_path)
    bucket_name = "tuning"
    object_name = dataset_path

    local_file_path = os.path.join("/tmp", os.path.basename(dataset_path))

    await async_minio_client.download_file(bucket_name, object_name, local_file_path)
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

            is_valid = await validate_dataset(
                decrypted_payload.dataset,
                decrypted_payload.dataset_type,
                decrypted_payload.file_format,
            )

            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid dataset format for {decrypted_payload.dataset_type} dataset type.",
                )
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

async def get_latest_model_submission(task_id: str):
    # TODO: Implement a proper lookup mechanism for model submissions
    try:
        # Placeholder: Return a fixed model name
        return 'cwaud/test'
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
