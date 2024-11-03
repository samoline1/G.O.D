import fastapi
from fastapi import HTTPException
from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer

from validator.core.config import Config


auth_scheme = HTTPBearer()


async def get_config(request: fastapi.Request) -> Config:
    config = request.app.state.config
    await config.psql_db.connect()

    return config

async def get_api_key(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if not credentials.credentials:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials
