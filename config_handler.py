import yaml
import os
import logging
from typing import Union

from const import CONFIG_TEMPLATE_PATH
from schemas import DatasetType, FileFormat, CustomDatasetType

logger = logging.getLogger(__name__)

def load_and_modify_config(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat
) -> dict:
    with open(CONFIG_TEMPLATE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['datasets'] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config['datasets'].append(dataset_entry)
    
    update_model_info(config, model)
    config['mlflow_experiment_name'] = dataset

    return config

def create_dataset_entry(
    dataset: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat
) -> dict:
    dataset_entry = {'path': dataset}

    if isinstance(dataset_type, DatasetType):
        dataset_entry['type'] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        custom_type_dict = {
            key: value for key, value in dataset_type.dict().items()
            if value is not None and (key != 'field' or value != "")
        }
        dataset_entry['type'] = custom_type_dict
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry['ds_type'] = file_format.value
        dataset_entry['data_files'] = [os.path.basename(dataset)]

    return dataset_entry

def update_model_info(config: dict, model: str):
    config['base_model'] = model

def save_config(config: dict, config_path: str):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

