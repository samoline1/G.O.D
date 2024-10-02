from transformers import AutoModel, AutoConfig
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.config import AxolotlConfig
from schemas import TrainRequest
from config_handler import create_dataset_entry, update_model_info

import yaml

def is_likely_finetune(original_repo: str, finetuned_model: AutoModel) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo)
    finetuned_config = finetuned_model.config
    attrs_to_compare = ['model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']
    architecture_same = all(getattr(original_config, attr) == getattr(finetuned_config, attr) for attr in attrs_to_compare)
    base_model_match = finetuned_config._name_or_path == original_repo
    has_peft_attributes = hasattr(finetuned_config, 'peft_config_path') or 'LoRA' in str(finetuned_config)
    has_lora_modules = any('lora' in name.lower() for name, _ in finetuned_model.named_modules())
    return architecture_same and (base_model_match or has_peft_attributes or has_lora_modules)

def evaluate_test_set_loss(config: AxolotlConfig):
    trainer = setup_trainer(config)
    eval_results = trainer.evaluate()
    return eval_results

def get_and_update_config(train_request: TrainRequest, config_path: str) -> AxolotlConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=train_request.dataset,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format
    )
    config_dict['datasets'] = [dataset_entry]
    update_model_info(config_dict, train_request.model)
    config = AxolotlConfig(**config_dict)
    return config

def perform_evaluation(train_request: TrainRequest, config_path: str):
    config = get_and_update_config(train_request, config_path)
    eval_results = evaluate_test_set_loss(config)
    return eval_results