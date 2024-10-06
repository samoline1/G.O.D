import os
from schemas import Job, DatasetType, FileFormat
from const import CONFIG_DIR, DOCKER_IMAGE, HUGGINGFACE_TOKEN
from configs.config_handler import load_and_modify_config, save_config
import docker
from docker.errors import DockerException
from const import CONFIG_DIR, OUTPUT_DIR
from utils import logger


def create_job(dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat) -> Job:
    return Job(
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format
    )

def process_job(job: Job):
    config_filename = f"{job.job_id}.yml"
    config_path = os.path.join(CONFIG_DIR, config_filename)

    config = load_and_modify_config(
        job.job_id,
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format
    )
    save_config(config, config_path)

    docker_env = {
        "HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN,
    }

    try:
        docker_client = docker.from_env()
        script_path = 'scripts/run_script_hf.sh' if job.file_format == FileFormat.HF else 'scripts/run_script_non_hf.sh'

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file not found: {script_path}")

        with open(script_path, 'r') as f:
            script_content = f.read()

        script_content = script_content.replace('{{JOB_ID}}', job.job_id)
        if job.file_format != FileFormat.HF:
            script_content = script_content.replace('{{DATASET_FILENAME}}', os.path.basename(job.dataset))

        volume_bindings = {
            os.path.abspath(CONFIG_DIR): {'bind': '/workspace/axolotl/configs', 'mode': 'rw'},
            os.path.abspath(OUTPUT_DIR): {'bind': '/workspace/axolotl/outputs', 'mode': 'rw'},
        }

        if job.file_format != FileFormat.HF:
            dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
            volume_bindings[dataset_dir] = {'bind': '/workspace/input_data', 'mode': 'ro'}

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

        if result['StatusCode'] != 0:
            raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

    except Exception as e:
        logger.error(f"Error processing job: {str(e)}")
        raise

    finally:
        if 'container' in locals():
            container.remove(force=True)

def stream_logs(container):
    log_buffer = ""
    for log_chunk in container.logs(stream=True, follow=True):
        try:
            log_buffer += log_chunk.decode('utf-8', errors='replace')
            while '\n' in log_buffer:
                line, log_buffer = log_buffer.split('\n', 1)
                logger.info(line.strip())
        except Exception as e:
            logger.error(f"Error processing log: {e}")
    
    if log_buffer:
        logger.info(log_buffer.strip())