import json
from typing import Any, List, AsyncGenerator
import httpx
import yaml
from datasets import load_dataset
from core.models.utility_models import Message, Role, Prompts
from fiber.logging_utils import get_logger
import asyncio
from validator.constants import PROMPT_PATH, ADDITIONAL_SYNTH_DATA_PERCENTAGE, PROMPT_GEN_ENDPOINT, PROMPT_GEN_TOKEN, SYNTH_GEN_BATCH_SIZE, SYNTH_MODEL_TEMPERATURE, SYNTH_MODEL
from validator.utils.call_endpoint import process_stream

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
    num_samples = int(train_dataset.num_rows * ADDITIONAL_SYNTH_DATA_PERCENTAGE)
    logger.info(f"Sampling {num_samples} samples from {dataset_name}")
    sampled_data = train_dataset.shuffle(seed=42).select(range(num_samples))
    sampled_data_list = [sample for sample in sampled_data]
    return sampled_data_list


def create_messages_from_row(row: dict, prompts: Prompts) -> List[Message]:
    messages = []
    system_message = Message(role=Role.SYSTEM, content=prompts.synth_data_creation_sys)
    messages.append(system_message)
    schema = json.dumps({key: value for key, value in row.items()})
    user_message = Message(
        role=Role.USER,
        content=prompts.synth_data_creation_prompt.format(schema=schema)
    )
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return set(synthetic_data_point.keys()) == set(original_data_columns)

async def generate_synthetic_dataset(sampled_data: List[dict]) -> List[dict]:
    prompts = load_prompts()
    logger.info(f"Creating synthetic dataset")
    synthetic_dataset = []

    async def process_row(row):
        messages = create_messages_from_row(row, prompts)
        payload = {
            "messages": [message.dict() for message in messages],
            "model": SYNTH_MODEL,
            "temperature": SYNTH_MODEL_TEMPERATURE,
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

    for i in range(0, len(sampled_data), SYNTH_GEN_BATCH_SIZE):
        batch = sampled_data[i:i + SYNTH_GEN_BATCH_SIZE]
        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        synthetic_dataset.extend([result for result in results if result is not None])

    return synthetic_dataset


