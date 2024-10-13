import os

import fastapi
from fastapi import HTTPException
from fastapi import Security
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer

from validator.core.config import Config


auth_scheme = HTTPBearer()


async def get_config(request: fastapi.Request) -> Config:
    config = request.app.state.config
    await config.psql_db.connect()  # FIXME: The pool does this, you shouldn't need to.

    return config


async def get_account_management_key(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    api_key = credentials.credentials
    expected_token = os.getenv("ACCOUNT_ACTIONS_API_KEY")  # FIXME: This should live in config and be set up on boot
    if api_key != expected_token:
        raise HTTPException(status_code=403, detail="Account management key invalid")
