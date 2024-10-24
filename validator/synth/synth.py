import asyncio
import json
from typing import List

import yaml
from datasets import load_dataset
from fiber.logging_utils import get_logger

from core.models.utility_models import Message
from core.models.utility_models import Prompts
from core.models.utility_models import Role
from validator.core.constants import ADDITIONAL_SYNTH_DATA_PERCENTAGE
from validator.core.constants import PROMPT_GEN_ENDPOINT
from validator.core.constants import PROMPT_GEN_TOKEN
from validator.core.constants import PROMPT_PATH
from validator.core.constants import SYNTH_GEN_BATCH_SIZE
from validator.core.constants import SYNTH_MODEL
from validator.core.constants import SYNTH_MODEL_TEMPERATURE
from validator.utils.call_endpoint import process_stream


logger = get_logger(__name__)


def load_prompts() -> Prompts:
    with open(PROMPT_PATH, "r") as file:
        prompts_dict = yaml.safe_load(file)
    return Prompts(**prompts_dict)


def load_and_sample_dataset(dataset_name: str, columns_to_sample: List[str]) -> List[dict]:
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logger.info('We needed to select the dataset config - assuming main')
        dataset = load_dataset(dataset_name, 'main')

    logger.info(f"Dataset: {dataset}")
    train_dataset = dataset["train"]
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
    user_message = Message(role=Role.USER, content=prompts.synth_data_creation_prompt.format(schema=schema))
    messages.append(user_message)
    return messages


def check_the_synthetic_data(synthetic_data_point: dict, original_data_columns: List[str]) -> bool:
    return set(synthetic_data_point.keys()) == set(original_data_columns)


async def generate_synthetic_dataset(sampled_data: List[dict]) -> List[dict]:
    prompts = load_prompts()
    logger.info("Creating synthetic dataset")
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
        except json.JSONDecodeError as e:
            logger.info('Json Eror was {e}')
            pass
        except Exception as e:
            logger.info('Erorr was {e}')
            pass
        return None  # Return None if there's an error or invalid data

    for i in range(0, len(sampled_data), SYNTH_GEN_BATCH_SIZE):
        batch = sampled_data[i : i + SYNTH_GEN_BATCH_SIZE]
        tasks = [process_row(row) for row in batch]
        results = await asyncio.gather(*tasks)
        if i < 2:
            logger.info(f"Coming out of synth is {results[0]}")
        synthetic_dataset.extend([result for result in results if result is not None])

    return synthetic_dataset
