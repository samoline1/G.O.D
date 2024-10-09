import os
import asyncio
from typing import List
from datasets import load_dataset, DatasetDict, Dataset
import datasets
from fiber.logging_utils import get_logger
from validator.synth.synth import generate_synthetic_dataset
import validator.constants as cst

logger = get_logger(__name__)

def train_test_split(dataset_name: str, test_size: float = None) -> DatasetDict:
    if test_size is None:
        test_size = cst.TRAIN_TEST_SPLIT_PERCENTAGE
    logger.info(f"Loading dataset '{dataset_name}'")
    dataset = load_dataset(dataset_name)
    
    if isinstance(dataset, DatasetDict):
        combined_dataset = datasets.concatenate_datasets([split for split in dataset.values()])
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

def upload_train_to_hf(train_dataset: Dataset, repo_name: str, token: str = None) -> None:
    logger.info(f"Uploading train dataset to Hugging Face Hub at '{repo_name}'")
    dataset_dict = DatasetDict({'train': train_dataset})
    dataset_dict.push_to_hub(repo_name, token=token, private=True)
    logger.info("Upload complete")

async def prepare_task(dataset_name: str, columns_to_sample: List[str], repo_name: str) -> Dataset:
    dataset_dict = train_test_split(dataset_name)
    train_dataset = dataset_dict['train']
    test_dataset = dataset_dict['test']
    
    synthetic_data = []
    if cst.GET_SYNTH_DATA:
        logger.info("Generating additional synthetic data")
        synthetic_data = await get_additional_synth_data(test_dataset, columns_to_sample)
        synthetic_dataset = datasets.Dataset.from_list(synthetic_data)
        logger.info("First 2 examples from original test dataset:")
        for i, example in enumerate(test_dataset.select(range(2))):
            logger.info(f"Example {i + 1}: {example}")
            
        logger.info("First 2 examples from synthetic dataset:")
        for i, example in enumerate(synthetic_dataset.select(range(2))):
            logger.info(f"Example {i + 1}: {example}")
    else:
        logger.info("Skipping synthetic data generation")

    upload_train_to_hf(train_dataset, repo_name, cst.HUGGINGFACE_TOKEN)
    return test_dataset, synthetic_data