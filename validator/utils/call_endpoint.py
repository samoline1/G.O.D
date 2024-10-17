import json
from typing import Any
import httpx
from collections.abc import AsyncGenerator
from typing import List

from fiber.logging_utils import get_logger
logger = get_logger(__name__)



async def process_stream(base_url: str, token: str, payload: dict[str, Any]) -> str:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    json_data = json.dumps(payload)
    logger.info(json_data)

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", base_url, content=json_data.encode('utf-8'), headers=headers) as response:
            response.raise_for_status()
            return ''.join([chunk async for chunk in _process_response(response)])

async def process_non_stream(base_url: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    json_data = json.dumps(payload)
    logger.info(f'The payload and url is {base_url}, {json_data}')
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(base_url, content=json_data.encode('utf-8'), headers=headers)
        response.raise_for_status()
        return response.json()

async def process_non_stream_get(base_url: str, token: str) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }

    logger.info(f'The GET request URL is {base_url}')
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
        except (IndexError, json.JSONDecodeError) as e:
            pass  # need to handle this

def _load_sse_jsons(chunk: str) -> List[dict[str, Any]]:
    return [json.loads(event.partition(":")[2]) for event in chunk.split("\n\n") if event and not event.startswith("data: [DONE]")]




