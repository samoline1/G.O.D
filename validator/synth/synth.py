import json
from typing import Any, List, AsyncGenerator
import httpx
import yaml
from core.constants import PROMPT_PATH, PERCENTAGE_SYNTH, PROMPT_GEN_ENDPOINT, PROMPT_GEN_TOKEN
from datasets import load_dataset
import random
from core.models.utility_models import Message, Role, Prompts
from fiber.logging_utils import get_logger
logger = get_logger(__name__)


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, 'r') as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)

def load_and_sample_dataset(dataset_name: str) -> List[dict]:
    dataset = load_dataset(dataset_name)
    logger.info(f"Dataset: {dataset}")
    
    train_dataset = dataset['train']
    
    num_samples = int(train_dataset.num_rows * PERCENTAGE_SYNTH)
    logger.info(f"Sampling {num_samples} samples from {dataset_name}")
    
    sampled_data = train_dataset.shuffle(seed=42).select(range(num_samples))
    
    sampled_data_list = [sample for sample in sampled_data]
    
    return sampled_data_list

async def process_stream(base_url: str, token: str, payload: dict[str, Any]) -> str:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    
    json_data = json.dumps(payload)
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", base_url, content=json_data.encode('utf-8'), headers=headers) as response:
            response.raise_for_status()
            return ''.join([chunk async for chunk in _process_response(response)])

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

def create_messages_from_row(row: dict, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.syth_data_creation_sys)
    messages.append(system_message)
    schema = json.dumps({key: value for key, value in row.items()})
    user_message = Message(role=Role.USER, content=prompts.syth_data_creation_prompt.format(schema=schema))
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return synthetic_data_point.keys() == original_data_columns

async def generate_synthetic_dataset(dataset_name: str) -> List[dict]:
    prompts = load_prompts()
    logger.info(f"Loading and sampling dataset: {dataset_name}")
    sampled_data = load_and_sample_dataset(dataset_name)
    logger.info(f"Creating synthetic dataset")
    synthetic_dataset = []

    for row in sampled_data:
        messages = create_messages_from_row(row, prompts)
        payload = {
            "messages": [message.dict() for message in messages],
            "model": "llama-3-1-70b",
            "temperature": 0.7,
        }
        try:
            synthetic_data_point = await process_stream(PROMPT_GEN_ENDPOINT, PROMPT_GEN_TOKEN, payload)
            json_synthetic_data_point = json.loads(synthetic_data_point)
            if check_the_synthetic_data(json_synthetic_data_point, row.keys()):
                synthetic_dataset.append(json_synthetic_data_point)
                logger.info(f"Synthetic data point added: {json_synthetic_data_point}")
        except json.JSONDecodeError:
            print(f"Error decoding synthetic data point: {synthetic_data_point}")
        except Exception as e:
            print(f"Error generating synthetic data point: {str(e)}")

    return synthetic_dataset


