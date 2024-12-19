import uuid

from fastapi import APIRouter
from fastapi import HTTPException
from fiber.logging_utils import get_logger

from core.models.payload_models import EvaluationRequest
from core.models.payload_models import EvaluationRequestDiffusion
from core.models.payload_models import EvaluationResult
from core.models.payload_models import EvaluationResultDiffusion
from core.models.payload_models import TaskRequest
from core.models.payload_models import TaskResponse
from core.models.utility_models import TaskStatus
from validator.evaluation.docker_evaluation import run_evaluation_docker
from validator.evaluation.docker_evaluation import run_evaluation_docker_diffusion
from validator.evaluation.task_prep import prepare_task


logger = get_logger(__name__)


async def evaluate_model(request: EvaluationRequest) -> EvaluationResult:
    if not request.dataset or not request.model or not request.original_model:
        raise HTTPException(status_code=400, detail="Dataset, model, original_model, and dataset_type are required.")

    try:
        eval_results = run_evaluation_docker(
            dataset=request.dataset,
            model=request.model,
            original_model=request.original_model,
            dataset_type=request.dataset_type,
            file_format=request.file_format,
        )
        return EvaluationResult(**eval_results.dict())
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def evaluate_model_diffusion(request: EvaluationRequestDiffusion) -> EvaluationResultDiffusion:
    try:
        eval_results = run_evaluation_docker_diffusion(
            test_split_path=request.test_split_path,
            base_model_repo=request.base_model_repo,
            base_model_filename=request.base_model_filename,
            lora_filename_list=request.lora_filename_list,
            lora_repo_list=request.lora_repo_list,
        )
        return EvaluationResultDiffusion(**eval_results.dict())
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def create_task(request: TaskRequest) -> TaskResponse:
    task_id = str(uuid.uuid4())
    columns = [request.system_col, request.instruction_col, request.input_col, request.output_col]
    test_dataset, synthetic_data = await prepare_task(request.ds_repo, columns, request.model_repo)
    # create a job in the database

    return TaskResponse(task_id=task_id, status=TaskStatus.PENDING)


async def get_task_status(task_id: str) -> TaskResponse:
    # get the task from the database
    return TaskResponse(task_id=task_id, status=TaskStatus.IDLE)


def factory():
    router = APIRouter()
    router.add_api_route("/evaluate/", evaluate_model, methods=["POST"])
    router.add_api_route("/create_task/", create_task, methods=["POST"])
    router.add_api_route("/get_task_status/", get_task_status, methods=["GET"])
    return router
