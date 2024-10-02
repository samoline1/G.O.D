from fastapi import APIRouter, HTTPException
from job_handler import create_job
from dataset_validator import validate_dataset
from schemas import TrainRequest, TrainResponse, JobStatusResponse, FileFormat

router = APIRouter()

@router.post("/train/", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    if not request.dataset or not request.model:
        raise HTTPException(status_code=400, detail="Dataset and model are required.")

    try:
        if request.file_format != FileFormat.HF:
            is_valid = validate_dataset(request.dataset, request.dataset_type, request.file_format)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid dataset format for {request.dataset_type} dataset type."
                )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job = create_job(
        dataset=request.dataset,
        model=request.model,
        dataset_type=request.dataset_type,
        file_format=request.file_format
    )
    router.worker.enqueue_job(job)

    return {"message": "Training job enqueued.", "job_id": job["job_id"]}

@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    status = router.worker.get_status(job_id)
    if status == "Not Found":
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JobStatusResponse(job_id=job_id, status=status)
