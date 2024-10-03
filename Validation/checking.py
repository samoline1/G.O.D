from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from schemas import TrainRequest
from config_handler import create_dataset_entry, update_model_info
import yaml
import json
import os
import logging
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from huggingface_hub import snapshot_download
from pathlib import Path
import tempfile
from torch.nn import CrossEntropyLoss
from datasets import Dataset
import torch

logger = logging.getLogger(__name__)

def is_likely_finetune(original_repo: str, finetuned_model: AutoModel) -> bool:
    original_config = AutoConfig.from_pretrained(original_repo)
    finetuned_config = finetuned_model.config
    
    if hasattr(finetuned_model, 'name_or_path'):
        finetuned_model_path = finetuned_model.name_or_path
    else:
        finetuned_model_path = finetuned_model.config._name_or_path
    
    adapter_config = os.path.join(finetuned_model_path, 'adapter_config.json')
    if os.path.exists(adapter_config):
        has_lora_modules = True
        logger.info(f"Adapter config found: {adapter_config}")
    else:
        logger.info(f"Adapter config not found at {adapter_config}")
        has_lora_modules = False
    logger.info(f"Original config: {original_config}")
    logger.info(f"Finetuned config: {finetuned_config}")
    attrs_to_compare = ['architectures', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads']
    for attr in attrs_to_compare:
        if getattr(original_config, attr) != getattr(finetuned_config, attr):
            logger.info(f"Original config: {getattr(original_config, attr)}")
            logger.info(f"Finetuned config: {getattr(finetuned_config, attr)}")
            logger.info(f"Attribute {attr} does not match")
    architecture_same = all(getattr(original_config, attr) == getattr(finetuned_config, attr) for attr in attrs_to_compare)
    base_model_match = finetuned_config._name_or_path == original_repo
    logger.info(f"Architecture same: {architecture_same}, Base model match: {base_model_match}, Has lora modules: {has_lora_modules}")
    return architecture_same and (base_model_match or has_lora_modules)

def get_and_update_config(train_request: TrainRequest, config_path: str) -> DictDefault:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=train_request.dataset,
        dataset_type=train_request.dataset_type,
        file_format=train_request.file_format
    )
    config_dict['datasets'] = [dataset_entry]
    update_model_info(config_dict, train_request.model)
    config = DictDefault(config_dict)
    return config

def perform_evaluation(train_request: TrainRequest, config_path: str, model: AutoModel, tokenizer: AutoTokenizer):
    config = get_and_update_config(train_request, config_path)
    eval_results = evaluate_test_set_loss(config, model, tokenizer)
    return eval_results


def evaluate_test_set_loss(cfg: DictDefault, model: AutoModel, tokenizer: AutoTokenizer):
    cfg = DictDefault(cfg)
    cfg.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {cfg}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ds_path = Path("data/")
        tmp_ds_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=cfg.datasets[0].path,
            repo_type="dataset",
            local_dir=tmp_ds_path,
        )

        prepared_path = Path(tmp_dir) / "prepared"
        dataset, _ = load_tokenized_prepared_datasets(
            tokenizer, cfg, prepared_path
        )

    eval_dataset = Dataset.from_dict({
        'input_ids': dataset['input_ids'],
        'attention_mask': dataset['attention_mask'],
        'labels': dataset['labels']
    })

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")
    from torch.nn import CrossEntropyLoss
    import numpy as np

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=1,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    predictions, labels, _ = trainer.predict(eval_dataset)

    logger.info(f"Predictions: {predictions}")
    logger.info(f"Labels: {labels}")

    logits_tensor = torch.tensor(predictions)
    labels_tensor = torch.tensor(labels)

    shift_logits = logits_tensor[..., :-1, :].contiguous()
    shift_labels = labels_tensor[..., 1:].contiguous()

    loss_fct = CrossEntropyLoss()
    eval_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

    predictions = np.argmax(predictions, axis=-1)
    accuracy = (predictions == labels).mean()

    eval_results = {
        "accuracy": accuracy,
        "eval_loss": eval_loss
    }

    logger.info(f"Evaluation results: {eval_results}")


    return eval_results