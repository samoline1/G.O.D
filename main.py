import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from miner.training_worker import TrainingWorker
from mpi.endpoints import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    worker = TrainingWorker()
    router.worker = worker
    app.state.worker = worker
    yield
    if app.state.worker:
        app.state.worker.shutdown()

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
