import yaml
from const import CONFIG_TEMPLATE_PATH, HUGGINGFACE_TOKEN
from schemas import DatasetType, FileFormat, CustomDatasetType
import os
import logging

logger = logging.getLogger(__name__)

def load_and_modify_config(job_id: str, dataset: str, model: str, dataset_type: Union[DatasetType, CustomDatasetType], file_format: FileFormat) -> dict:
    with open(CONFIG_TEMPLATE_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['datasets'] = []

    dataset_entry = {'path': dataset}

    if isinstance(dataset_type, DatasetType):
        dataset_entry['type'] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        dataset_entry['type'] = {
            'system_prompt': dataset_type.system_prompt or "",
            'system_format': dataset_type.system_format or "{system}",
            'field_system': dataset_type.field_system or "",
            'field_instruction': dataset_type.field_instruction or "",
            'field_input': dataset_type.field_input or "",
            'field_output': dataset_type.field_output or "",
            'format': dataset_type.format or "",
            'no_input_format': dataset_type.no_input_format or "",
            'field': dataset_type.field or ""
        }
    else:
        raise ValueError("Invalid dataset_type provided.")

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

