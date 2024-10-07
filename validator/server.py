from fastapi import APIRouter, HTTPException
from core.utils import validate_dataset
from core.models.utility_models import FileFormat
from core.models.payload_models import EvaluationRequest, EvaluationResponse
from fiber.logging_utils import get_logger
import docker
import json
import os
from miner.logic.job_handler import stream_logs
import core.constants as cst

logger = get_logger(__name__)

router = APIRouter()

async def evaluate_model(request: EvaluationRequest) -> EvaluationResponse:
    if not request.dataset or not request.model or not request.original_model:
        raise HTTPException(
            status_code=400, detail="Dataset, model, and original_model are required."
        )

    try:
        if request.file_format != FileFormat.HF:
            is_valid = validate_dataset(
                request.dataset, request.dataset_type, request.file_format
            )
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid dataset format for {request.dataset_type} dataset type.",
                )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    client = docker.from_env()

    environment = {
        "DATASET": request.dataset,
        "MODEL": request.model,
        "ORIGINAL_MODEL": request.original_model,
        "DATASET_TYPE": request.dataset_type.value if isinstance(request.dataset_type, DatasetType) else "custom",
        "FILE_FORMAT": request.file_format.value,
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
    }

    try:
        container = client.containers.run(
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[['gpu']])],
            detach=True,
        )

        stream_logs(container)

        result = container.wait()
        logs = container.logs().decode("utf-8")
        container.remove()

        if result["StatusCode"] != 0:
            raise Exception(f"Evaluation failed: {logs}")

        eval_results = json.loads(logs)
        return EvaluationResponse(**eval_results)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def factory():
    router = APIRouter()
    router.add_api_route("/evaluate/", evaluate_model, methods=["POST"])
    return router
