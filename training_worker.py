import threading
import queue
import uuid
import os
import docker
from docker.errors import DockerException
import logging
from schemas import DatasetType, FileFormat, JobStatus
from const import CONFIG_DIR, OUTPUT_DIR, DOCKER_IMAGE, HUGGINGFACE_TOKEN
from config_handler import load_and_modify_config, save_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
            "HUGGINGFACE_TOKEN": HUGGINGFACE_TOKEN,
        }

        logger.debug(f"HUGGINGFACE_TOKEN: {HUGGINGFACE_TOKEN}")

        try:
            logger.info(f"Running Docker container with dataset: {job.dataset}")

            script_path = 'scripts/run_script_hf.sh' if job.file_format == FileFormat.HF else 'scripts/run_script_non_hf.sh'
            logger.debug(f"Selected script path: {script_path}")

            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script file not found: {script_path}")

            with open(script_path, 'r') as f:
                script_content = f.read()

            logger.debug(f"Original script content:\n{script_content}")

            script_content = script_content.replace('{{JOB_ID}}', job.job_id)
            if job.file_format != FileFormat.HF:
                script_content = script_content.replace('{{DATASET_FILENAME}}', os.path.basename(job.dataset))

            logger.debug(f"Modified script content:\n{script_content}")

            volume_bindings = {
                os.path.abspath(CONFIG_DIR): {'bind': '/workspace/axolotl/configs', 'mode': 'rw'},
                os.path.abspath(OUTPUT_DIR): {'bind': '/workspace/axolotl/outputs', 'mode': 'rw'},
            }

            if job.file_format != FileFormat.HF:
                dataset_dir = os.path.dirname(os.path.abspath(job.dataset))
                volume_bindings[dataset_dir] = {'bind': '/workspace/input_data', 'mode': 'ro'}

            logger.debug(f"Volume bindings: {volume_bindings}")

            logger.info("Starting Docker container...")
            container = self.docker_client.containers.run(
                image=DOCKER_IMAGE,
                command=["/bin/bash", "-c", script_content],
                volumes=volume_bindings,
                environment=docker_env,
                runtime="nvidia",
                detach=True,
                tty=True,
            )
            logger.info(f"Docker container started with ID: {container.id}")

            log_thread = threading.Thread(target=self.stream_logs, args=(container,))
            log_thread.start()

            result = container.wait()
            log_thread.join()

            if result['StatusCode'] != 0:
                raise DockerException(f"Container exited with non-zero status code: {result['StatusCode']}")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            self.update_job_status(job, JobStatus.FAILED, str(e))
            raise e
        except DockerException as e:
            logger.error(f"Docker exception occurred: {e}")
            self.update_job_status(job, JobStatus.FAILED, str(e))
            raise e
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            self.update_job_status(job, JobStatus.FAILED, str(e))
            raise e
        finally:
            if 'container' in locals():
                container.remove(force=True)

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