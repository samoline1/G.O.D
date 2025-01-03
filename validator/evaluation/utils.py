import base64
import os
import shutil
import tempfile
from io import BytesIO

import numpy as np
from datasets import get_dataset_config_names
from fiber.logging_utils import get_logger
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig
from transformers import AutoModelForCausalLM


logger = get_logger(__name__)


def model_is_a_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo, token=os.environ.get("HUGGINGFACE_TOKEN"))
    finetuned_config = finetuned_model.config

    try:
        if hasattr(finetuned_model, "name_or_path"):
            finetuned_model_path = finetuned_model.name_or_path
        else:
            finetuned_model_path = finetuned_model.config._name_or_path

        adapter_config = os.path.join(finetuned_model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            has_lora_modules = True
            logger.info(f"Adapter config found: {adapter_config}")
        else:
            logger.info(f"Adapter config not found at {adapter_config}")
            has_lora_modules = False
        base_model_match = finetuned_config._name_or_path == original_repo
    except Exception as e:
        logger.debug(f"There is an issue with checking the finetune path {e}")
        base_model_match = True
        has_lora_modules = False

    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "n_layer",
        "model_type",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = True
    for attr in attrs_to_compare:
        if hasattr(original_config, attr):
            if not hasattr(finetuned_config, attr):
                architecture_same = False
                break
            if getattr(original_config, attr) != getattr(finetuned_config, attr):
                architecture_same = False
                break

    logger.info(
        f"Architecture same: {architecture_same}, Base model match: {base_model_match}, Has lora modules: {has_lora_modules}"
    )
    return architecture_same and (base_model_match or has_lora_modules)


def get_default_dataset_config(dataset_name: str) -> str | None:
    try:
        logger.info(dataset_name)
        config_names = get_dataset_config_names(dataset_name)
    except Exception:
        return None
    if config_names:
        logger.info(f"Taking the first config name: {config_names[0]} for dataset: {dataset_name}")
        # logger.info(f"Dataset {dataset_name} has configs: {config_names}. Taking the first config name: {config_names[0]}")
        return config_names[0]
    else:
        return None


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str) -> str:
    # Use a temp folder to ensure correct file placement
    try:
        local_filename = os.path.basename(filename)
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                shutil.move(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
        return final_path
    except Exception as e:
        logger.error(f"Error downloading file: {e}")


def calculate_l2_loss(test_image: Image.Image, generated_image: Image.Image) -> float:
    test_image = np.array(test_image.convert("RGB")) / 255.0
    generated_image = np.array(generated_image.convert("RGB")) / 255.0
    if test_image.shape != generated_image.shape:
        raise ValueError("Images must have the same dimensions to calculate L2 loss.")
    l2_loss = np.mean((test_image - generated_image) ** 2)
    return l2_loss
