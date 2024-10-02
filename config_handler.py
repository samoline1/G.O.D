import yaml
from const import CONFIG_TEMPLATE_PATH, HUGGINGFACE_TOKEN
from schemas import DatasetType, FileFormat
import os

import logging

logger = logging.getLogger(__name__)

def load_and_modify_config(job_id: str, dataset: str, model: str, dataset_type: DatasetType, file_format: FileFormat) -> dict:
    with open(CONFIG_TEMPLATE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['datasets'] = []

    dataset_entry = {
        'path': dataset,
        'type': dataset_type.value
    }

    if file_format != FileFormat.HF:
        dataset_entry['ds_type'] = file_format.value
        dataset_entry['data_files'] = [os.path.basename(dataset)]
    else:
        pass  # No additional keys needed

    config['datasets'].append(dataset_entry)

    config['base_model'] = model

    return config

def save_config(config: dict, config_path: str):
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

