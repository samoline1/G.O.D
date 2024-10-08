from functools import partial
from fastapi.routing import APIRouter
from fiber.miner.dependencies import blacklist_low_stake, get_config, verify_request
from fiber.miner.security.encryption import decrypt_general_payload
from core.models.payload_models import CapacityPayload
from fiber.logging_utils import get_logger
from fastapi import Depends, Header
from fiber.miner.core.configuration import Config
from fiber import constants as fcst

logger = get_logger(__name__)


async def capacity(
    configs: CapacityPayload = Depends(
        partial(decrypt_general_payload, CapacityPayload)
    ),
    validator_hotkey: str = Header(..., alias=fcst.VALIDATOR_HOTKEY),
    config: Config = Depends(get_config),
) -> dict[str, float | str]:
    return {}


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/capacity",
        capacity,
        tags=["Subnet"],
        methods=["POST"],
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    )
    return router
