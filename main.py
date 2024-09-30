from fastapi import FastAPI
from training_worker import TrainingWorker
from endpoints import router

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    worker = TrainingWorker()
    router.worker = worker
    app.state.worker = worker

@app.on_event("shutdown")
async def shutdown_event():
    if app.state.worker:
        app.state.worker.shutdown()

app.include_router(router)
