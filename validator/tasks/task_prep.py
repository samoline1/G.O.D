import json
import os
import tempfile
from typing import List

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from fiber.logging_utils import get_logger

import validator.core.constants as cst
from validator.synth.synth import generate_synthetic_dataset
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)


async def save_json_to_temp_file(data: List[dict], prefix: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", prefix=prefix)
    with open(temp_file.name, "w") as f:
        json.dump(data, f)
    return temp_file.name


async def upload_json_to_minio(file_path: str, bucket_name: str, object_name: str) -> str:
    await async_minio_client.upload_file(bucket_name, object_name, file_path)
    return await async_minio_client.get_presigned_url(bucket_name, object_name)


def train_test_split(dataset_name: str, test_size: float = None) -> DatasetDict:
    if test_size is None:
        test_size = cst.TRAIN_TEST_SPLIT_PERCENTAGE
    logger.info(f"Loading dataset '{dataset_name}'")
    try:
        dataset = load_dataset(dataset_name)
    except:
        logger.info('Assuming main name on a failure')
        dataset = load_dataset(dataset_name, 'main')

    if isinstance(dataset, DatasetDict):
        combined_dataset = concatenate_datasets([split for split in dataset.values()])
    else:
        combined_dataset = dataset

    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    logger.info(f"Splitting combined dataset into train and test with test size {test_size}")

    split_dataset = combined_dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    logger.info(f"Train set size: {len(split_dataset['train'])}")
    logger.info(f"Test set size: {len(split_dataset['test'])}")
    return split_dataset


async def get_additional_synth_data(dataset: Dataset, columns_to_sample: List[str]) -> List[dict]:
    num_samples = min(cst.MAX_SYNTH_DATA_POINTS, int(len(dataset) * cst.ADDITIONAL_SYNTH_DATA_PERCENTAGE))
    logger.info(f"Generating {num_samples} additional synthetic data points")
    sampled_data = dataset.shuffle(seed=42).select(range(num_samples))
    sampled_data = sampled_data.remove_columns([col for col in sampled_data.column_names if col not in columns_to_sample])
    sampled_data_list = [sample for sample in sampled_data]
    synthetic_data = await generate_synthetic_dataset(sampled_data_list)
    return synthetic_data


def change_to_json_format(dataset: Dataset, columns: List[str]):
    logger.info(f"HERE  ARE THE COLUMNS {columns}")
    return [{col: row[col] for col in columns} for row in dataset]


async def prepare_task(dataset_name: str, columns_to_sample: List[str]) -> tuple[str, str, str]:
    logger.info(f"Preparing {dataset_name}")
    dataset_dict = train_test_split(dataset_name)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]

    synthetic_data = []
    if cst.GET_SYNTH_DATA:
        logger.info("Generating additional synthetic data")
        synthetic_data = await get_additional_synth_data(test_dataset, columns_to_sample)
        synthetic_dataset = Dataset.from_list(synthetic_data)
        logger.info("First 2 examples from original test dataset:")
        for i, example in enumerate(test_dataset.select(range(2))):
            logger.info(f"Example {i + 1}: {example}")

        logger.info("First 2 examples from synthetic dataset:")
        for i, example in enumerate(synthetic_dataset.select(range(2))):
            logger.info(f"Example {i + 1}: {example}")
    else:
        logger.info("Skipping synthetic data generation")

    # this looks ugly
    train_data_json = change_to_json_format(train_dataset, columns_to_sample)
    test_data_json = change_to_json_format(test_dataset, columns_to_sample)
    synthetic_data_json = change_to_json_format(synthetic_data, columns_to_sample) if synthetic_data else []

    train_json_path = await save_json_to_temp_file(train_data_json, prefix="train_data_")
    test_json_path = await save_json_to_temp_file(test_data_json, prefix="test_data_")
    synth_json_path = await save_json_to_temp_file(synthetic_data_json, prefix="synth_data_") if synthetic_data else None

    train_json_url = await upload_json_to_minio(train_json_path, "tuning", f"{dataset_name}_train_data.json")
    test_json_url = await upload_json_to_minio(test_json_path, "tuning", f"{dataset_name}_test_data.json")
    synth_json_url = (
        await upload_json_to_minio(synth_json_path, "tuning", f"{dataset_name}_synth_data.json") if synthetic_data else None
    )

    os.remove(test_json_path)
    if synth_json_path:
        os.remove(synth_json_path)

    return test_json_url.strip('"'), synth_json_url.strip('"'), train_json_url.strip('"')
