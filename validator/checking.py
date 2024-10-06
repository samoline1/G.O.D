from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from schemas import TrainRequest
from configs.config_handler import create_dataset_entry, update_model_info
from const import VALI_CONFIG_PATH
import yaml
import os
from utils import logger
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def is_likely_finetune(original_repo: str, finetuned_model: AutoModelForCausalLM) -> bool:
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

def perform_evaluation(train_request: TrainRequest, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    config = get_and_update_config(train_request, VALI_CONFIG_PATH)
    eval_results = evaluate_test_set_loss(config, model, tokenizer)
    return eval_results

def evaluate_test_set_loss(cfg: DictDefault, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    cfg = DictDefault(cfg)
    cfg.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {cfg}")

    eval_dataset = load_evaluation_dataset(cfg, tokenizer)
    log_dataset_and_model_info(eval_dataset, model, tokenizer)
    eval_dataloader = create_eval_dataloader(eval_dataset, cfg, tokenizer)

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    total_loss, num_batches = process_evaluation_batches(model, eval_dataloader, device)

    eval_results = calculate_evaluation_results(total_loss, num_batches)
    logger.info(f"Final evaluation results: {eval_results}")

    return eval_results

def load_evaluation_dataset(cfg: DictDefault, tokenizer: AutoTokenizer):
    prepared_path = Path(cfg.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(tokenizer, cfg, prepared_path)
    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
    return eval_dataset

def log_dataset_and_model_info(eval_dataset, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model config: {model.config}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {model.config.vocab_size}")

def create_eval_dataloader(eval_dataset, cfg: DictDefault, tokenizer: AutoTokenizer):
    return DataLoader(
        eval_dataset,
        batch_size=cfg.micro_batch_size,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        shuffle=False
    )

def collate_fn(batch, tokenizer: AutoTokenizer):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def process_evaluation_batches(model: AutoModelForCausalLM, eval_dataloader: DataLoader, device: torch.device):
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            logger.info(f"Processing batch {batch_idx + 1}")
            loss = compute_batch_loss(model, batch, device)
            logger.info(f"Batch {batch_idx + 1} loss: {loss}")
            total_loss += loss
            num_batches += 1

    return total_loss, num_batches

def compute_batch_loss(model: AutoModelForCausalLM, batch: dict, device: torch.device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1), 
                           ignore_index=-100)

    return loss.item()

def calculate_evaluation_results(total_loss: float, num_batches: int):
    if num_batches > 0:
        average_loss = total_loss / num_batches
        logger.info(f"Average loss: {average_loss}")
    else:
        logger.error("No valid batches were processed during evaluation.")
        average_loss = float('inf')

    return {
        "eval_loss": average_loss,
        "perplexity": torch.exp(torch.tensor(average_loss)).item()
    }