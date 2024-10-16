import os
from uuid import UUID
import yaml
import docker
from docker.errors import DockerException
from core.models.utility_models import Job, DatasetType, FileFormat, CustomDatasetType
from core import constants as cst
from core.config.config_handler import create_dataset_entry, save_config, update_model_info
from fiber.logging_utils import get_logger
from core.docker_utils import stream_logs

logger = get_logger(__name__)

def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
    task_id: UUID
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    update_model_info(config, model, task_id)
    config["mlflow_experiment_name"] = dataset

    return config

def create_job(
        task_id: UUID, dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat
) -> Job:
    return Job(
        task_id=task_id, dataset=dataset, model=model, dataset_type=dataset_type, file_format=file_format
    )

def start_tuning_container(job: Job):
    config_filename = f"{job.task_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.dataset, job.model, job.dataset_type, job.file_format, job.task_id
    )
    save_config(config, config_path)

    docker_env = {
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
        "WANDB_TOKEN": cst.WANDB_TOKEN,
        "JOB_ID": job.task_id,
        "DATASET_TYPE": job.dataset_type.value if isinstance(job.dataset_type, DatasetType) else "custom",
        "DATASET_FILENAME": os.path.basename(job.dataset) if job.file_format != FileFormat.HF else "",
    }
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()

        volume_bindings = {
            os.path.abspath(cst.CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(cst.OUTPUT_DIR): {
                "bind": "/workspace/axolotl/outputs",
                "mode": "rw",
            },
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            volume_bindings[dataset_dir] = {
                "bind": "/workspace/input_data",
                "mode": "ro",
            }

        container = docker_client.containers.run(
            image=cst.MINER_DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[['gpu']])],
            detach=True,
            tty=True,
        )

        # Use the shared stream_logs function
        stream_logs(container)

        result = container.wait()

        if result["StatusCode"] != 0:
            raise DockerException(
                f"Container exited with non-zero status code: {result['StatusCode']}"
            )

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if "container" in locals():
            container.remove(force=True)
