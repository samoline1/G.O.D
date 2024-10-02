import threading
import queue
import uuid
import os
import docker
from docker.errors import DockerException
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from huggingface_hub import create_repo, upload_folder
import hashlib
import tempfile
from schemas import DatasetType, FileFormat, JobStatus
from const import CONFIG_DIR, OUTPUT_DIR, COMPLETED_MODEL_DIR, DOCKER_IMAGE, HUGGINGFACE_TOKEN, WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, REPO, USR
from config_handler import load_and_modify_config, save_config

import shlex

class TrainingJob:
    def __init__(self, dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat):
        self.dataset = dataset
        self.model = model
        self.dataset_type = dataset_type
        self.file_format = file_format
        self.job_id = str(uuid.uuid4())
        self.status = JobStatus.QUEUED

class TrainingWorker:
    def __init__(self):
        self.job_queue = queue.Queue()
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        self.job_status = {}
        self.status_lock = threading.Lock()
        self.docker_client = docker.from_env()
        self.docker_client.ping()

    def worker(self):
        while True:
            job: TrainingJob = self.job_queue.get()
            if job is None:
                break
            self.update_job_status(job, JobStatus.RUNNING)
            try:
                self.process_job(job)
                self.update_job_status(job, JobStatus.COMPLETED)
            except Exception as e:
                self.update_job_status(job, JobStatus.FAILED, str(e))
            finally:
                self.job_queue.task_done()

    def update_job_status(self, job: TrainingJob, status: JobStatus, error_message: str = None):
        with self.status_lock:
            job.status = status
            self.job_status[job.job_id] = status.value
            if error_message:
                self.job_status[job.job_id] += f": {error_message}"

    def process_job(self, job: TrainingJob):
        os.makedirs(CONFIG_DIR, exist_ok=True)
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
            "WANDB_API_KEY": WANDB_API_KEY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_ENTITY": WANDB_ENTITY,
            "HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN,
        }

        try:
            logger.info(f"Running Docker container with dataset: {job.dataset}")

            # Create a temporary shell script
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.sh') as temp_script:
                temp_script.write(f"""#!/bin/bash
set -ex
env | grep -E 'HUGGINGFACE_TOKEN|WANDB'
mkdir -p /workspace/axolotl/data/
cp /workspace/input_data/{shlex.quote(os.path.basename(job.dataset))} /workspace/axolotl/data/{shlex.quote(os.path.basename(job.dataset))}
echo 'Data copied successfully'
pip install mlflow
echo 'MLflow installed successfully'
echo 'Logging into Hugging Face registry'
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Attempting to log in to Hugging Face"
    echo "$HUGGINGFACE_TOKEN" | huggingface-cli login --token-stdin
else
    echo "HUGGINGFACE_TOKEN is not set. Skipping login."
fi
echo 'Starting training command'
accelerate launch -m axolotl.cli.train /workspace/axolotl/configs/{shlex.quote(job.job_id)}.yml
""")
                temp_script_path = temp_script.name

            # Set execute permissions for the script
            os.chmod(temp_script_path, 0o755)

            volume_bindings = {
                os.path.abspath(CONFIG_DIR): {'bind': '/workspace/axolotl/configs', 'mode': 'rw'},
                os.path.abspath(OUTPUT_DIR): {'bind': '/workspace/axolotl/outputs', 'mode': 'rw'},
                os.path.dirname(os.path.abspath(job.dataset)): {'bind': '/workspace/input_data', 'mode': 'rw'},
                temp_script_path: {'bind': '/tmp/run_script.sh', 'mode': 'ro'},
            }

            container = self.docker_client.containers.run(
                image=DOCKER_IMAGE,
                command=["/bin/bash", "/tmp/run_script.sh"],
                volumes=volume_bindings,
                environment=docker_env,
                runtime="nvidia",
                detach=True,
                tty=True,
            )

            log_thread = threading.Thread(target=self.stream_logs, args=(container,))
            log_thread.start()

            result = container.wait()
            log_thread.join()

            if result['StatusCode'] != 0:
                raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

        except DockerException as e:
            logger.error(f"Docker exception occurred: {e}")
            self.update_job_status(job, JobStatus.FAILED, str(e))
            raise e
        finally:
            container.remove(force=True)
            os.unlink(temp_script_path)  # Remove the temporary script

    def stream_logs(self, container):
        out = ""
        for log_chunk in container.logs(stream=True, follow=True):
            try:
                out += log_chunk.decode('utf-8', errors='replace').strip()
            except Exception as e:
                logger.error(f"Error decoding log chunk: {e}")
        logger.info(out)

    def enqueue_job(self, job: TrainingJob):
        self.update_job_status(job, JobStatus.QUEUED)
        self.job_queue.put(job)

    def get_status(self, job_id: str) -> str:
        with self.status_lock:
            return self.job_status.get(job_id, "Not Found")

    def shutdown(self):
        self.job_queue.put(None)
        self.thread.join()
        self.docker_client.close()