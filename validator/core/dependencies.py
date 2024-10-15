
import fastapi
from fastapi.security import HTTPBearer

from validator.core.config import Config


auth_scheme = HTTPBearer()


async def get_config(request: fastapi.Request) -> Config:
    config = request.app.state.config
    await config.psql_db.connect()

    return config
