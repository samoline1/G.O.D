import os
import yaml
import docker
from docker.errors import DockerException
from core.models.utility_models import Job, DatasetType, FileFormat, CustomDatasetType
from core import constants as cst
from core.config.config_handler import create_dataset_entry, save_config, update_model_info
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:
    with open(cst.CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    update_model_info(config, model)
    config["mlflow_experiment_name"] = dataset

    return config

def create_job(
    dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat
) -> Job:
    return Job(
        dataset=dataset, model=model, dataset_type=dataset_type, file_format=file_format
    )

def start_tuning_container(job: Job):
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(cst.CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.dataset, job.model, job.dataset_type, job.file_format
    )
    save_config(config, config_path)

    docker_env = {
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
        "JOB_ID": job.job_id,
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
            image=cst.DOCKER_IMAGE,
            environment=docker_env,
            volumes=volume_bindings,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[['gpu']])],
            detach=True,
            tty=True,
        )

        _stream_logs(container)

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

def _stream_logs(container):
    log_buffer = ""
    for log_chunk in container.logs(stream=True, follow=True):
        try:
            log_buffer += log_chunk.decode("utf-8", errors="replace")
            while "\n" in log_buffer:
                line, log_buffer = log_buffer.split("\n", 1)
                logger.info(line.strip())
        except Exception as e:
            logger.error(f"Error processing log: {e}")

    if log_buffer:
        logger.info(log_buffer.strip())