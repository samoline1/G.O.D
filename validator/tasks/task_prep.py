import json
import os
import tempfile
import asyncio
from typing import List

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from fiber.logging_utils import get_logger

import validator.core.constants as cst
from validator.evaluation.utils import get_default_dataset_config
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
        config_name = get_default_dataset_config(dataset_name)
        dataset = load_dataset(dataset_name, config_name)
    except Exception as e:
        logger.exception(f'Failed to load dataset {dataset_name}: {e}')
        raise e

    if isinstance(dataset, DatasetDict):
        combined_dataset = concatenate_datasets([split for split in dataset.values()])
    else:
        combined_dataset = dataset

    logger.info(f"Combined dataset size: {len(combined_dataset)}")
    logger.info(f"Splitting combined dataset into train and test with test size {test_size}")

    test_size = min(int(len(combined_dataset) * cst.TRAIN_TEST_SPLIT_PERCENTAGE), cst.MAX_SYNTH_DATA_POINTS)
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

async def process_batch_dict(batch: list, columns: List[str], batch_num: int) -> List[dict]:
    logger.info(f"Processing batch {batch_num}, batch type: {type(batch)}")
    if batch:
        logger.info(f"First item in batch type: {type(batch[0])}")

    batch_json = []
    for idx, row in enumerate(batch):
        try:
            if isinstance(row, dict):
                row_dict = {col: row.get(col, '') for col in columns}
            else:
                logger.warning(f"Unexpected row type in batch {batch_num}: {type(row)}")
                row_dict = {col: '' for col in columns}
            batch_json.append(row_dict)

            # Log first item of each batch
            if idx == 0:
                logger.info(f"Batch {batch_num} first item: {row_dict}")

        except Exception as e:
            logger.error(f"Error processing row in batch {batch_num}: {e}")
            logger.error(f"Problematic row: {row}")
            row_dict = {col: '' for col in columns}
            batch_json.append(row_dict)

    return batch_json

async def change_to_json_format_async(dataset: Dataset | list, columns: List[str], batch_size: int = 1000):
    logger.info(f"Input dataset type: {type(dataset)}")

    if isinstance(dataset, list):
        logger.info("Converting list to Dataset")
        try:
            dataset = Dataset.from_list(dataset)
        except Exception as e:
            logger.info(f"Could not convert to Dataset, proceeding with list. Error: {e}")
            # If we can't convert to Dataset, we'll process the list directly
            total_rows = len(dataset)
    else:
        total_rows = len(dataset)

    total_batches = (total_rows + batch_size - 1) // batch_size
    logger.info(f"Starting processing of {total_rows} rows in {total_batches} batches")
    logger.info(f"Columns to extract: {columns}")

    tasks = []
    for i in range(0, total_rows, batch_size):
        batch_num = i // batch_size
        end_idx = min(i + batch_size, total_rows)
        logger.info(f"Creating batch {batch_num} ({i}:{end_idx})")

        try:
            if isinstance(dataset, Dataset):
                batch = dataset.select(range(i, end_idx))
            else:
                # If we're working with a list, slice it directly
                batch = dataset[i:end_idx]

            tasks.append(process_batch_dict(batch, columns, batch_num))

        except Exception as e:
            logger.error(f"Error creating batch {batch_num}: {e}")
            continue

    logger.info(f"Created {len(tasks)} batch processing tasks")

    processed_batches = await asyncio.gather(*tasks)
    logger.info("All batches processed, combining results")

    result = []
    for batch in processed_batches:
        result.extend(batch)

    logger.info(f"Processing complete. Total items in result: {len(result)}")
    if result:
        logger.info(f"Sample from final result: {result[0]}")

    return result
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
    train_data_json = await change_to_json_format_async(train_dataset, columns_to_sample)
    test_data_json = await change_to_json_format_async(test_dataset, columns_to_sample)
    synthetic_data_json = await change_to_json_format_async(synthetic_data, columns_to_sample) if synthetic_data else []

    train_json_path = await save_json_to_temp_file(train_data_json, prefix="train_data_")
    test_json_path = await save_json_to_temp_file(test_data_json, prefix="test_data_")
    synth_json_path = await save_json_to_temp_file(synthetic_data_json, prefix="synth_data_") if synthetic_data else None

    train_json_url = await upload_json_to_minio(train_json_path, "tuning", f"{os.urandom(8).hex()}_train_data.json")
    test_json_url = await upload_json_to_minio(test_json_path, "tuning", f"{os.urandom(8).hex()}_test_data.json")
    synth_json_url = (
        await upload_json_to_minio(synth_json_path, "tuning", f"{os.urandom(8).hex()}_synth_data.json") if synthetic_data else None
    )

    os.remove(test_json_path)
    if synth_json_path:
        os.remove(synth_json_path)

    return test_json_url.strip('"'), synth_json_url.strip('"'), train_json_url.strip('"')
