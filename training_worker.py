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
            "WANDB_LOGIN_MODE": "silent",

        }

        try:
            logger.info(f"Running Docker container with dataset: {job.dataset}")

            dataset_filename = os.path.basename(job.dataset)

            mkdir_command = "mkdir -p /workspace/axolotl/data/"
            copy_command = (
                f"cp /workspace/input_data/{dataset_filename} "
                f"/workspace/axolotl/data/{dataset_filename}"
            )
            install_mlflow_command = "pip install mlflow"
            training_command = (
                f"accelerate launch -m axolotl.cli.train /workspace/axolotl/configs/{job.job_id}.yml"
            )

            full_command = (
                f"/bin/bash -c 'set -x && "
                f"env | grep -E \"HUGGINGFACE_TOKEN|WANDB\" && "
                f"{mkdir_command} && echo \"Directory created\" && "
                f"{copy_command} && echo \"File copied successfully\" && "
                f"ls -la /workspace/axolotl/data/ && "
                f"{install_mlflow_command} && echo \"MLflow installed successfully\" && "
                f"echo \"Starting training command\" && "
                f"{training_command} || echo \"Training command failed with exit code $?\"'"
            )

            logger.info(f"Command to be executed: {full_command}")

            container = self.docker_client.containers.run(
                image=DOCKER_IMAGE,
                command=full_command,
                volumes={
                    os.path.dirname(os.path.abspath(job.dataset)): {'bind': '/workspace/input_data', 'mode': 'rw'},
                    os.path.abspath(CONFIG_DIR): {'bind': '/workspace/axolotl/configs', 'mode': 'rw'},
                    os.path.abspath(OUTPUT_DIR): {'bind': '/workspace/axolotl/outputs', 'mode': 'rw'},
                },
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

    def stream_logs(self, container):
            out = ""
            for log_chunk in container.logs(stream=True, follow=True):
                try:
                   out += log_chunk.decode('utf-8', errors='replace').strip()
                except Exception as e:
                    logger.error(f"Error decoding log chunk: {e}")
            logger.info(out)

    def compute_model_hash(self, model_dir: str) -> str:
        hash_sha256 = hashlib.sha256()
        for root, dirs, files in os.walk(model_dir):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def upload_model_to_hf(self, job_id: str):
        output_dir = os.path.join(OUTPUT_DIR, COMPLETED_MODEL_DIR)
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Trained model directory '{output_dir}' does not exist.")
        
        model_hash = self.compute_model_hash(output_dir)
        repo_name = f"{REPO}/{USR}_{model_hash}"

        try:
            create_repo(repo_id=repo_name, token=self.hf_token, private=True)
        except Exception as e:
            print(f"Error creating repository {repo_name}: {e}")
            raise e

        with tempfile.TemporaryDirectory() as temp_dir:
            upload_folder(
                folder_path=output_dir,
                repo_id=repo_name,
                repo_type="model",
                token=HUGGINGFACE_TOKEN,
                ignore_patterns=["*.tmp", "*.bak", ".git*"],
            )

        print(f"Model successfully uploaded to {repo_name}")

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
