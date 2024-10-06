import os
from core.models.utility_models import Job, DatasetType, FileFormat
from const import CONFIG_DIR, DOCKER_IMAGE, HUGGINGFACE_TOKEN
from api.configs.config_handler import create_dataset_entry, save_config, update_model_info
import docker
from docker.errors import DockerException
from const import OUTPUT_DIR
from utils import logger
import yaml

from const import CONFIG_TEMPLATE_PATH
from core.models.utility_models import CustomDatasetType




# TODO: give a much nicer name - maybe even a docstring
# I have no idea what this is referring to or doing
def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:
    with open(CONFIG_TEMPLATE_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    update_model_info(config, model)
    config["mlflow_experiment_name"] = dataset

    return config

#  TODO: Much nicer names please
def create_job(
    dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat
) -> Job:
    return Job(
        dataset=dataset, model=model, dataset_type=dataset_type, file_format=file_format
    )

# TODO: Dutty code
def process_job(job: Job):
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)

    config = _load_and_modify_config(
        job.job_id, job.dataset, job.model, job.dataset_type, job.file_format
    )
    save_config(config, config_path)

    docker_env = {
        "HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN,
    }
    logger.info(f"Docker environment: {docker_env}")

    try:
        docker_client = docker.from_env()
        script_path = (
            "scripts/run_script_hf.sh"
            if job.file_format == FileFormat.HF
            else "scripts/run_script_non_hf.sh"
        )
        logger.info(f"Selected script path: {script_path}")

        if not os.path.exists(script_path):
            logger.error(f"Script file not found: {script_path}")
            raise FileNotFoundError(f"Script file not found: {script_path}")

        logger.info(f"Script file exists: {script_path}")

        with open(script_path, "r") as f:
            script_content = f.read()
            logger.info(f"Original script content:\n{script_content}")

        script_content = script_content.replace("{{JOB_ID}}", job.job_id)
        if job.file_format != FileFormat.HF:
            script_content = script_content.replace(
                "{{DATASET_FILENAME}}", os.path.basename(job.dataset)
            )
            logger.info(f"Modified script content:\n{script_content}")

        volume_bindings = {
            os.path.abspath(CONFIG_DIR): {
                "bind": "/workspace/axolotl/configs",
                "mode": "rw",
            },
            os.path.abspath(OUTPUT_DIR): {
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
        logger.info(f"Starting container with script: {script_content}")

        container = docker_client.containers.run(
            image=DOCKER_IMAGE,
            command=["/bin/bash", "-c", script_content],
            volumes=volume_bindings,
            environment=docker_env,
            runtime="nvidia",
            detach=True,
            tty=True,
        )

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

# Whats this for, wen typehints?
def stream_logs(container):
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
