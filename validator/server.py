from fastapi import APIRouter, HTTPException
from core.utils import validate_dataset
from core.models.utility_models import (
    FileFormat,
)
from core.models.payload_models import EvaluationRequest, EvaluationResponse
from validator.evaluation import utils as eval_utils
from validator.evaluation import eval
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import logger

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

    finetuned_model = AutoModelForCausalLM.from_pretrained(request.model)
    tokenizer = AutoTokenizer.from_pretrained(request.original_model)
    is_finetune = eval_utils.model_is_a_finetune(
        request.original_model, finetuned_model
    )

    if not is_finetune:
        logger.info(
            "The provided model does not appear to be a fine-tune of the original model."
        )
        # TODO: So what? What do we do with it?

    eval_results = eval.evaluate_finetuned_model(request, finetuned_model, tokenizer)

    return EvaluationResponse(is_finetune=is_finetune, eval_results=eval_results)


def factory():
    router = APIRouter()
    router.add_api_route("/evaluate/", evaluate_model, methods=["POST"])
