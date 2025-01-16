from validator.tasks.task_prep import prepare_text_task
from core.models.utility_models import TaskStatus
from datasets import get_dataset_infos
from fiber import Keypair
from validator.core.models import ImageRawTask
from validator.core.models import TextRawTask
from core.models.payload_models import TrainRequestDiffusion
from core.models.payload_models import TrainRequestText
from core.models.utility_models import CustomDatasetType
from core.models.utility_models import FileFormat
from validator.utils.logging import create_extra_log
from validator.utils.logging import logger


def get_total_text_dataset_size(repo_name: str) -> int:
    return int(sum(info.dataset_size for info in get_dataset_infos(repo_name).values() if info.dataset_size))


# TODO - how best to implement this - should we not save this when we create the task and have it as an attribute?
def get_total_image_dataset_size(repo_name: str) -> int:
    return 100


async def run_image_task_prep(task: ImageRawTask, keypair: Keypair) -> ImageRawTask:
    task.status = TaskStatus.LOOKING_FOR_NODES


#    raw_data = await download_s3_file(task.ds)
# implement me as you have in your testing in terms of how you would expect this
# after you've downloaded, just need to split it and then reupload as a train and test


async def run_text_task_prep(task: TextRawTask, keypair: Keypair) -> TextRawTask:
    columns_to_sample = [
        i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
    ]
    test_data, synth_data, train_data = await prepare_text_task(
        dataset_name=task.ds, columns_to_sample=columns_to_sample, keypair=keypair
    )
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data
    logger.info(
        "Data creation is complete - now time to find some miners",
        extra=create_extra_log(status=task.status),
    )
    return task

def prepare_text_task_request(task: TextRawTask) -> TrainRequestText:
    dataset_type = CustomDatasetType(
        field_system=task.field_system,
        field_input=task.field_input,
        field_output=task.field_output,
        field_instruction=task.field_instruction,
        format=task.format,
        no_input_format=task.no_input_format,
    )

    dataset = task.training_data if task.training_data else "dataset error"
    task_request_body = TrainRequestText(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
    )

    return task_request_body


# probs better to be Image not Diffusion for consistency
def prepare_image_task_request(task: ImageRawTask) -> TrainRequestDiffusion:
    # TODO MISSING STUFF THAT I DON'T KNOW ABOUT
    return TrainRequestDiffusion(
        model=task.model_filename,
        task_id=task.task_id,
        hours_to_complete=task.hours_to_complete,
        dataset_zip=task.ds,
        # where do we get this from in the task definition?
        hf_repo="not sure what this is",
        hf_folder="or this",
    )

