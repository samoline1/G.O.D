import threading
import queue
import docker
from core.models.utility_models import Job, JobStatus
from utils import logger
from miner.logic.job_handler import process_job


class TrainingWorker:
    def __init__(self):
        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        self.docker_client = docker.from_env()

    def worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                process_job(job)
                job.status = JobStatus.COMPLETED
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
            finally:
                self.job_queue.task_done()

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def get_status(self, job_id: str) -> JobStatus:
        job = self.job_store.get(job_id)
        return job.status if job else "Not Found"

    def shutdown(self):
        self.job_queue.put(None)
        self.thread.join()
        self.docker_client.close()
