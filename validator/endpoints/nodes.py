from fastapi import APIRouter
from fastapi import Body
from fastapi import Depends
from loguru import logger

from validator.core.config import Config
from validator.core.dependencies import get_config
from validator.db import sql


async def add_node(
    coldkey: str = Body(..., embed=True),
    ip: str = Body(..., embed=True),
    ip_type: str = Body(..., embed=True),
    port: int = Body(..., embed=True),
    symmetric_key: str = Body(..., embed=True),
    network: float = Body(..., embed=True),
    stake: float = Body(..., embed=True),
    config: Config = Depends(get_config),
):
    node_id = await sql.add_node(coldkey, ip, ip_type, port, symmetric_key, network, stake, config.psql_db)

    logger.info(f"Node {node_id} added.")
    return {"success": True, "node_id": node_id}


def factory_router() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/nodes/add",
        add_node,
        tags=["nodes"],
        methods=["POST"],
    )

    return router
