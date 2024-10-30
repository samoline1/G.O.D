import json
from collections.abc import AsyncGenerator
from typing import Any
from typing import List, Optional

from fiber.networking.models import NodeWithFernet as Node
from fiber.validator import client

import httpx
from fiber.logging_utils import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from validator.core.config import Config


logger = get_logger(__name__)

# Create a retry decorator with exponential backoff
retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True
)

async def process_non_stream_fiber_get(endpoint: str, config: Config, node: Node) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    logger.info(f"Attempting to hit a GET {server_address} endpoint {endpoint}")
    logger.info(f"With his keypair {config.keypair} this fernet {node.fernet} this key {node.symmetric_key_uuid}")
    assert node.symmetric_key_uuid is not None
    try:
        response = await client.make_non_streamed_get(
            httpx_client=config.httpx_client,
            server_address=server_address,
            validator_ss58_address=config.keypair.ss58_address,
            symmetric_key_uuid=node.symmetric_key_uuid,
            endpoint=endpoint,
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Failed to comunication with node {node.node_id}: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"Failed to communicate with node {node.node_id}")
        return None

    return response.json()


async def process_non_stream_fiber(endpoint: str, config: Config, node: Node, payload: dict[str, Any]) -> dict[str, Any] | None:
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )
    logger.info(f"Attempting to hit {server_address} endpoint {endpoint} with payload {payload}")
    logger.info(f"With his keypair {config.keypair} this fernet {node.fernet} this key {node.symmetric_key_uuid}")
    assert node.symmetric_key_uuid is not None
    try:
        response = await client.make_non_streamed_post(
            httpx_client=config.httpx_client,
            server_address=server_address,
            validator_ss58_address=config.keypair.ss58_address,
            miner_ss58_address=node.hotkey,
            keypair=config.keypair,
            fernet=node.fernet,
            symmetric_key_uuid=node.symmetric_key_uuid,
            endpoint=endpoint,
            payload=payload,
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Failed to comunication with node {node.node_id}: {e}")
        return None

    if response.status_code != 200:
        logger.warning(f"Failed to communicate with node {node.node_id}")
        return None

    return response.json()

@retry_with_backoff
async def process_stream(base_url: str, token: str, payload: dict[str, Any]) -> str:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    json_data = json.dumps(payload)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", base_url, content=json_data.encode("utf-8"), headers=headers) as response:
            response.raise_for_status()
            return "".join([chunk async for chunk in _process_response(response)])


# If this it to talk to the miner, its already in fiber
# We can change to that once we add bittensor stuff (i know that's why its like this ATM)
@retry_with_backoff
async def process_non_stream_get(base_url: str, token: Optional[str]) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.get(base_url, headers=headers)
        response.raise_for_status()
        return response.json()


async def _process_response(response: httpx.Response) -> AsyncGenerator[str, None]:
    async for line in response.aiter_lines():
        try:
            loaded_jsons = _load_sse_jsons(line)
            for text_json in loaded_jsons:
                content = text_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    yield content
        except (IndexError, json.JSONDecodeError):
            pass  # need to handle this


def _load_sse_jsons(chunk: str) -> List[dict[str, Any]]:
    return [
        json.loads(event.partition(":")[2]) for event in chunk.split("\n\n") if event and not event.startswith("data: [DONE]")
    ]
