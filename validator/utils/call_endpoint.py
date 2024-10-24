import json
from collections.abc import AsyncGenerator
from typing import Any
from typing import List, Optional

import httpx
from fiber.logging_utils import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = get_logger(__name__)

# Create a retry decorator with exponential backoff
retry_with_backoff = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    reraise=True
)


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


@retry_with_backoff
async def process_non_stream(base_url: str, token: Optional[str], payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    json_data = json.dumps(payload)

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(base_url, content=json_data.encode("utf-8"), headers=headers)
        response.raise_for_status()
        return response.json()


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
