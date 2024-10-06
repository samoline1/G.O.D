from transformers import AutoModelForCausalLM, AutoConfig
import os
from utils import logger


def model_is_a_finetune(
    original_repo: str, finetuned_model: AutoModelForCausalLM
) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo)
    finetuned_config = finetuned_model.config

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
    logger.info(f"Original config: {original_config}")
    logger.info(f"Finetuned config: {finetuned_config}")
    attrs_to_compare = [
        "architectures",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    architecture_same = all(
        getattr(original_config, attr) == getattr(finetuned_config, attr)
        for attr in attrs_to_compare
    )
    base_model_match = finetuned_config._name_or_path == original_repo
    logger.info(
        f"Architecture same: {architecture_same}, Base model match: {base_model_match}, Has lora modules: {has_lora_modules}"
    )
    return architecture_same and (base_model_match or has_lora_modules)
