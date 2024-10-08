import json
from typing import Any, List, AsyncGenerator
import httpx
import yaml
from core.constants import PROMPT_PATH, PERCENTAGE_SYNTH, PROMPT_GEN_ENDPOINT, PROMPT_GEN_TOKEN
from datasets import load_dataset
from core.models.utility_models import Message, Role, Prompts
from fiber.logging_utils import get_logger
import asyncio
from core.constants import SYNTH_BATCH_SIZE, SYNTH_TEMPERATURE, SYNTH_MODEL

logger = get_logger(__name__)


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, 'r') as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)

def load_and_sample_dataset(dataset_name: str, columns_to_sample: List[str]) -> List[dict]:
    dataset = load_dataset(dataset_name)
    logger.info(f"Dataset: {dataset}")
    train_dataset = dataset['train']
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_sample])
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

async def generate_synthetic_dataset(dataset_name: str, columns_to_sample: List[str]) -> List[dict]:
    prompts = load_prompts()
    logger.info(f"Loading and sampling dataset: {dataset_name}")
    sampled_data = load_and_sample_dataset(dataset_name, columns_to_sample)
    logger.info(f"Creating synthetic dataset")
    synthetic_dataset = []

    async def process_row(row):
        messages = create_messages_from_row(row, prompts)
        payload = {
            "messages": [message.dict() for message in messages],
            "model": SYNTH_MODEL,
            "temperature": SYNTH_TEMPERATURE,
        }
        try:
            synthetic_data_point = await process_stream(PROMPT_GEN_ENDPOINT, PROMPT_GEN_TOKEN, payload)
            json_synthetic_data_point = json.loads(synthetic_data_point)
            if check_the_synthetic_data(json_synthetic_data_point, row.keys()):
                return json_synthetic_data_point
        except json.JSONDecodeError:
            logger.error(f"Error decoding synthetic data point: {synthetic_data_point}")
        except Exception as e:
            logger.error(f"Error generating synthetic data point: {str(e)}")
        return None  # Return None if there's an error or invalid data

    for i in range(0, len(sampled_data), SYNTH_BATCH_SIZE):
        batch = sampled_data[i:i + SYNTH_BATCH_SIZE]
        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        logger.info(f"Additional synthetic data points: {results}")
        synthetic_dataset.extend([result for result in results if result is not None])

    return synthetic_dataset


