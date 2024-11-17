from fiber.logging_utils import get_logger
from fiber.miner import server

from miner.endpoints.tuning import factory_router as tuning_factory_router


logger = get_logger(__name__)

# BIT annoyign as i want to extend the lifespan. how? - NTS
app = server.factory_app(debug=True)


tuning_router = tuning_factory_router()

app.include_router(tuning_router)

# if os.getenv("ENV", "prod").lower() == "dev":
#    configure_extra_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7999)
