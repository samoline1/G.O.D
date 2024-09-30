from contextlib import asynccontextmanager
from fastapi import FastAPI
from training_worker import TrainingWorker
from endpoints import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    worker = TrainingWorker()
    router.worker = worker
    app.state.worker = worker
    yield
    # Shutdown
    if app.state.worker:
        app.state.worker.shutdown()

app = FastAPI(lifespan=lifespan)
app.include_router(router)
