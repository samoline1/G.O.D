from typing import Union, Dict
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import json
from core.config.config_handler import create_dataset_entry, update_model_info
from core import constants as cst
from core.models.utility_models import CustomDatasetType, DatasetType, FileFormat
from fiber.logging_utils import get_logger
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault

logger = get_logger(__name__)

def _load_and_update_evaluation_config(
    dataset_name: str,
    language_model: AutoModelForCausalLM,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat,
    config_path: str,
) -> DictDefault:
    """Load and update the configuration for model evaluation."""
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    dataset_entry = create_dataset_entry(
        dataset=dataset_name,
        dataset_type=dataset_type,
        file_format=file_format,
    )
    config_dict["datasets"] = [dataset_entry]
    update_model_info(config_dict, language_model)
    return DictDefault(config_dict)


def _load_evaluation_dataset(
    evaluation_config: DictDefault, tokenizer: AutoTokenizer
) -> Dataset:
    """Load the evaluation dataset."""
    prepared_path = Path(evaluation_config.output_dir) / "prepared"
    eval_dataset, _ = load_tokenized_prepared_datasets(
        tokenizer, evaluation_config, prepared_path
    )
    logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} samples")
    return eval_dataset


def _log_dataset_and_model_info(
    eval_dataset: Dataset,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> None:
    """Log information about the dataset and model."""
    logger.info(f"Eval dataset sample: {eval_dataset[0]}")
    logger.info(f"Model type: {type(language_model)}")
    logger.info(f"Model config: {language_model.config}")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    logger.info(f"Model vocabulary size: {language_model.config.vocab_size}")


def _create_evaluation_dataloader(
    eval_dataset: Dataset, evaluation_config: DictDefault, tokenizer: AutoTokenizer
) -> DataLoader:
    """Create a DataLoader for the evaluation dataset."""
    return DataLoader(
        eval_dataset,
        batch_size=evaluation_config.micro_batch_size,
        collate_fn=lambda batch: _collate_evaluation_batch(batch, tokenizer),
        shuffle=False,
    )


def _collate_evaluation_batch(
    batch: list[dict[str, list[int]]], tokenizer: AutoTokenizer
) -> dict[str, torch.Tensor]:
    """Collate function for batching dataset items."""
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]

    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _process_evaluation_batches(
    language_model: AutoModelForCausalLM,
    eval_dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, int]:
    """Process evaluation batches and compute total loss."""
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            logger.info(f"Processing batch {batch_idx + 1}")
            batch_loss = _compute_batch_loss(language_model, batch, device)
            logger.info(f"Batch {batch_idx + 1} loss: {batch_loss}")
            total_loss += batch_loss
            num_batches += 1

    return total_loss, num_batches


def _compute_batch_loss(
    language_model: AutoModelForCausalLM, batch: dict, device: torch.device
) -> float:
    """Compute the loss for a single batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = language_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss.item()


def _calculate_evaluation_metrics(
    total_loss: float, num_batches: int
) -> dict[str, float]:
    """Calculate evaluation metrics based on total loss and number of batches."""
    if num_batches > 0:
        average_loss = total_loss / num_batches
        logger.info(f"Average loss: {average_loss}")
    else:
        logger.error("No valid batches were processed during evaluation.")
        average_loss = float("inf")

    return {
        "eval_loss": average_loss,
        "perplexity": torch.exp(torch.tensor(average_loss)).item(),
    }




def evaluate_language_model_loss(
    evaluation_config: DictDefault,
    language_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, float]:
    """Evaluate the loss of a language model on a test set."""
    evaluation_config.tokenizer_config = tokenizer.name_or_path
    logger.info(f"Config: {evaluation_config}")

    eval_dataset = _load_evaluation_dataset(evaluation_config, tokenizer)
    _log_dataset_and_model_info(eval_dataset, language_model, tokenizer)
    eval_dataloader = _create_evaluation_dataloader(
        eval_dataset, evaluation_config, tokenizer
    )

    language_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    language_model.to(device)

    total_loss, num_batches = _process_evaluation_batches(
        language_model, eval_dataloader, device
    )

    evaluation_results = _calculate_evaluation_metrics(total_loss, num_batches)
    logger.info(f"Final evaluation results: {evaluation_results}")

    return evaluation_results




def evaluate_finetuned_model() -> dict[str, float]:
    dataset = os.environ["DATASET"]
    model = os.environ["MODEL"]
    original_model = os.environ["ORIGINAL_MODEL"]
    dataset_type = os.environ["DATASET_TYPE"]
    file_format = os.environ["FILE_FORMAT"]

    finetuned_model, tokenizer = load_model_and_tokenizer()
    
    is_finetune = model_is_a_finetune(original_model, finetuned_model)
    
    if not is_finetune:
        logger.warning("The provided model does not appear to be a fine-tune of the original model.")

    evaluation_config = _load_and_update_evaluation_config(
        dataset, finetuned_model, dataset_type, file_format, cst.VALI_CONFIG_PATH
    )
    
    eval_results = evaluate_language_model_loss(evaluation_config, finetuned_model, tokenizer)
    
    results = {
        "is_finetune": is_finetune,
        "eval_results": eval_results
    }
    
    print(json.dumps(results))
    return results

def load_model_and_tokenizer():
    model_name = os.environ["MODEL"]
    original_model_name = os.environ["ORIGINAL_MODEL"]
    
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return model, tokenizer

def model_is_a_finetune(original_model: str, finetuned_model: AutoModelForCausalLM) -> bool:
    original_config = AutoConfig.from_pretrained(original_model)
    finetuned_config = finetuned_model.config

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
    
    adapter_config = os.path.join(finetuned_model.name_or_path, "adapter_config.json")
    has_lora_modules = os.path.exists(adapter_config)
    
    base_model_match = finetuned_config._name_or_path == original_model
    
    return architecture_same and (base_model_match or has_lora_modules)

if __name__ == "__main__":
    results = evaluate_finetuned_model()
    print(results)