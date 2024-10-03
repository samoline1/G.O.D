from transformers import AutoModel, AutoConfig, AutoTokenizer
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
import shutil
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

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
    import torch
    from torch.nn import CrossEntropyLoss
    from tqdm import tqdm

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

    logger.info(f"Dataset: {dataset}")

    # Set the model to evaluation mode
    model.eval()

    # Initialize loss function
    loss_fn = CrossEntropyLoss()

    # Initialize variables to store total loss and number of samples
    total_loss = 0.0
    total_samples = 0

    # Get the maximum sequence length for the model
    max_length = model.config.max_position_embeddings

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        # Iterate through the entire dataset
        for batch in tqdm(dataset, desc="Evaluating"):
            # Convert lists to tensors
            input_ids = torch.tensor(batch['input_ids'])
            attention_mask = torch.tensor(batch['attention_mask'])
            labels = torch.tensor(batch['labels'])

            # Truncate or pad sequences to match the model's maximum length
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
            labels = labels[:, :max_length]

            # Pad sequences if they're shorter than max_length
            if input_ids.shape[1] < max_length:
                padding_length = max_length - input_ids.shape[1]
                input_ids = torch.nn.functional.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length), value=0)
                labels = torch.nn.functional.pad(labels, (0, padding_length), value=-100)  # -100 is typically ignored in loss calculation

            # Move tensors to the model's device
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Update total loss and sample count
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)

    # Calculate average loss
    avg_loss = total_loss / total_samples

    logger.info(f"Evaluation completed. Average loss: {avg_loss:.4f}")

    return {"loss": avg_loss}