from fastapi import Depends, HTTPException

from core.models import payload_models
from fastapi.routing import APIRouter
from fiber.logging_utils import get_logger

from core.models.payload_models import JobStatusResponse, TrainResponse
from core.models.utility_models import FileFormat, JobStatus
from core.utils import validate_dataset
from fiber.miner.core.configuration import Config
from fiber.miner.dependencies import get_config
from miner.config import WorkerConfig
from miner.dependencies import get_worker_config
from miner.logic.job_handler import create_job

logger = get_logger(__name__)


async def tune_model(
    decrypted_payload: payload_models.TrainRequest,
    config: Config = Depends(get_config),
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    if not decrypted_payload.dataset or not decrypted_payload.model:
        raise HTTPException(status_code=400, detail="Dataset and model are required.")

    try:
        if decrypted_payload.file_format != FileFormat.HF:
            is_valid = validate_dataset(
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
        dataset=decrypted_payload.dataset,
        model=decrypted_payload.model,
        dataset_type=decrypted_payload.dataset_type,
        file_format=decrypted_payload.file_format,
    )
    worker_config.trainer.enqueue_job(job)

    return {"message": "Training job enqueued.", "job_id": job.job_id}


async def get_job_status(
    decrypted_payload: payload_models.JobStatusPayload,
    worker_config: WorkerConfig = Depends(get_worker_config),
):
    status = worker_config.trainer.get_status(decrypted_payload.job_id)
    if status == JobStatus.NOT_FOUND:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JobStatusResponse(job_id=decrypted_payload.job_id, status=status)


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/train/",
        tune_model,
        tags=["Subnet"],
        methods=["POST"],
        # dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
        response_model=TrainResponse,
    )
    router.add_api_route(
        "/status/{job_id}",
        get_job_status,
        tags=["Subnet"],
        methods=["GET"],
        # dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
        response_model=JobStatusResponse,
    )
    return router
