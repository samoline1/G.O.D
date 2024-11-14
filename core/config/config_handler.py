import os

import yaml
from fiber.logging_utils import get_logger
from transformers import AutoTokenizer

from core.models.utility_models import CustomDatasetType
from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat


logger = get_logger(__name__)


def create_dataset_entry(
    dataset: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:
    """
    Create a dataset entry for the configuration file.

    Args:
        dataset (str): The path or identifier of the dataset.
        dataset_type (DatasetType | CustomDatasetType): The type of the dataset,
        a simple example would be INSTRUCT which will have columns [INSTRUCTION, INPUT, OUTPUT]
        file_format (FileFormat): The format of the dataset file - HF, JSON, CSV -
        HF is a dataset on Hugging Face, JSON and CSV are local files

    Returns:
        dict: A dictionary containing the dataset entry information,
        this is basically the dataset information, how it should be parsed by the axolotl library
    """

    dataset_entry = {"path": dataset}

    if file_format == FileFormat.JSON:
        dataset_entry = {"path": f"/workspace/input_data/{os.path.basename(dataset)}"}

    if isinstance(dataset_type, DatasetType):
        dataset_entry["type"] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        custom_type_dict = {key: value for key, value in dataset_type.model_dump().items() if value is not None}
        dataset_entry["format"] = "custom"
        dataset_entry["type"] = custom_type_dict
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def update_model_info(config: dict, model: str, job_id: str = ""):
    logger.info("WE ARE UPDATING THE MODEL INFO")

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # we need to make sure the pad token is defined
    logger.info(f"HERE ARE THE CURRENT PAD TOKENS STUFF {tokenizer.pad_token} {tokenizer.eos_token}")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        config["special_tokens"] = {"pad_token": tokenizer.eos_token}

    config["huggingface_username"] = config.get("HUGGINGFACE_USERNAME")
    if not config["huggingface_username"]:
        raise ValueError("Environment variable HUGGINGFACE_USERNAME is not set. Make sure to run create_config.py.")

    config["base_model"] = model
    config["wandb_runid"] = job_id
    config["wandb_name"] = job_id
    config["hub_model_id"] = f"{config['huggingface_username']}/{job_id}"
    return config


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)
