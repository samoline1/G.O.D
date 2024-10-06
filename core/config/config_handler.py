import yaml
import os

from core.models.utility_models import DatasetType, FileFormat, CustomDatasetType

# TODO: docstring or smth with this - no idea what its doing
def create_dataset_entry(
    dataset: str,
    dataset_type: DatasetType | CustomDatasetType,
    file_format: FileFormat,
) -> dict:
    dataset_entry = {"path": dataset}

    if isinstance(dataset_type, DatasetType):
        dataset_entry["type"] = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        custom_type_dict = {
            key: value
            for key, value in dataset_type.model_dump().items()
            if value is not None and (key != "field" or value != "")
        }
        dataset_entry["type"] = custom_type_dict
    else:
        raise ValueError("Invalid dataset_type provided.")

    if file_format != FileFormat.HF:
        dataset_entry["ds_type"] = file_format.value
        dataset_entry["data_files"] = [os.path.basename(dataset)]

    return dataset_entry


def update_model_info(config: dict, model: str):
    config["base_model"] = model


def save_config(config: dict, config_path: str):
    with open(config_path, "w") as file:
        yaml.dump(config, file)
