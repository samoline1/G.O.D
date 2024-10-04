from transformers import AutoModel, AutoConfig, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from schemas import TrainRequest
from config_handler import create_dataset_entry, update_model_info
import yaml
import os
import logging
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.trainer import setup_trainer
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

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

from transformers import TrainerCallback
from axolotl.utils.callbacks import LossWatchDogCallback
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict


class LossExtractorCallback(TrainerCallback):
    def __init__(self):
        self.eval_loss = None

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if 'eval_loss' in metrics:
            self.eval_loss = metrics['eval_loss']

def evaluate_test_set_loss(cfg: DictDefault, model: AutoModel, tokenizer: AutoTokenizer):
    cfg = DictDefault(cfg)
    cfg.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {cfg}")

    prepared_path = Path(cfg.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(
        tokenizer, cfg, prepared_path
    )

    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Labels in sample: {eval_dataset[0]['labels']}")

    # Ensure that the dataset has labels
    if 'labels' not in eval_dataset[0]:
        raise ValueError("The dataset does not contain 'labels'. Please check your data preparation step.")

    # Custom collate function to handle potentially nested labels
    def collate_fn(batch):
        logger.info(f"Collating batch of size {len(batch)}")
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        
        # Log the structure of labels before processing
        logger.info(f"Labels before processing: {[item['labels'] for item in batch]}")
        
        # Flatten labels if they're nested
        labels = [torch.tensor(item['labels']) if isinstance(item['labels'][0], int) 
                  else torch.tensor([l for sublist in item['labels'] for l in sublist]) 
                  for item in batch]
        
        # Log the structure of labels after processing
        logger.info(f"Labels after processing: {labels}")

        # Pad sequences
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is often used to ignore padding in loss calculation

        logger.info(f"Padded input_ids shape: {input_ids.shape}")
        logger.info(f"Padded attention_mask shape: {attention_mask.shape}")
        logger.info(f"Padded labels shape: {labels.shape}")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    # Create DataLoader with custom collate function
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.micro_batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    logger.info(f"Created DataLoader with batch size {cfg.micro_batch_size}")

    # Ensure model is in evaluation mode
    model.eval()
    logger.info(f"Model set to evaluation mode")

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            logger.info(f"Processing batch {batch_idx + 1}")
            # Move batch to the same device as the model
            batch = {k: v.to(device) for k, v in batch.items()}
            
            logger.info(f"Batch input_ids shape: {batch['input_ids'].shape}")
            logger.info(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
            logger.info(f"Batch labels shape: {batch['labels'].shape}")
            
            outputs = model(**batch)
            loss = outputs.loss
            
            if loss is not None:
                logger.info(f"Batch {batch_idx + 1} loss: {loss.item()}")
                total_loss += loss.item()
                num_batches += 1
            else:
                logger.warning(f"Loss is None for batch {batch_idx + 1}")

    if num_batches > 0:
        average_loss = total_loss / num_batches
        logger.info(f"Average loss: {average_loss}")
    else:
        logger.error("No valid batches were processed during evaluation.")
        average_loss = float('inf')

    eval_results = {
        "eval_loss": average_loss,
        "perplexity": torch.exp(torch.tensor(average_loss)).item()
    }

    logger.info(f"Eval results: {eval_results}")

    return eval_results