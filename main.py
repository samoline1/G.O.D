import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import queue
from endpoints import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    job_queue = queue.Queue()
    job_store = {}
    router.job_queue = job_queue
    router.job_store = job_store
    app.state.job_queue = job_queue
    app.state.job_store = job_store
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
