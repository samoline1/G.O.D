import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import queue
import threading
from endpoints import router
from training_worker import process_job

def worker(job_queue, job_store):
    while True:
        job = job_queue.get()
        if job is None:
            break
        try:
            process_job(job)
            job_store[job['job_id']]['status'] = 'Completed'
        except Exception as e:
            job_store[job['job_id']]['status'] = 'Failed'
            job_store[job['job_id']]['error'] = str(e)
        finally:
            job_queue.task_done()

@asynccontextmanager
async def lifespan(app: FastAPI):
    job_queue = queue.Queue()
    job_store = {}
    worker_thread = threading.Thread(target=worker, args=(job_queue, job_store), daemon=True)
    worker_thread.start()
    
    router.job_queue = job_queue
    router.job_store = job_store
    app.state.job_queue = job_queue
    app.state.job_store = job_store
    yield
    job_queue.put(None)  # Signal the worker to stop
    worker_thread.join()

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
