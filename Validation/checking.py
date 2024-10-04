from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Trainer, TrainingArguments
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

logger = logging.getLogger(__name__)

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

def perform_evaluation(train_request: TrainRequest, config_path: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    config = get_and_update_config(train_request, config_path)
    eval_results = evaluate_test_set_loss(config, model, tokenizer)
    return eval_results

def evaluate_test_set_loss(cfg: DictDefault, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
    cfg = DictDefault(cfg)
    cfg.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {cfg}")

    prepared_path = Path(cfg.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(
        tokenizer, cfg, prepared_path
    )

    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(model)}")
    logger.info(f"Model config: {model.config}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {model.config.vocab_size}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_eval_batch_size=cfg.micro_batch_size,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        num_train_epochs=1,  # Not used for evaluation, but required
        report_to="none",  # Disable wandb or other integrations
    )

    trainer = setup_trainer(
        cfg,
        model=model,
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    eval_loss = eval_results.get("eval_loss")
    perplexity = torch.exp(torch.tensor(eval_loss)).item() if eval_loss is not None else None

    final_results = {
        "eval_loss": eval_loss,
        "perplexity": perplexity
    }

    logger.info(f"Final evaluation results: {final_results}")

    return final_results