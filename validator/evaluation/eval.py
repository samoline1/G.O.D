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
from core.models.utility_models import CustomDatasetType, DatasetType, FileFormat
from fiber.logging_utils import get_logger
from axolotl.utils.data import load_tokenized_prepared_datasets
from axolotl.utils.dict import DictDefault
from core.models.payload_models import EvaluationResult
import json
from datasets import load_dataset
from validator.evaluation.utils import model_is_a_finetune

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

    results = {
        "average_loss": average_loss,
        "perplexity": 2 ** average_loss,
    }

    with open('/app/results/evaluation_results.json', 'w') as f:
        json.dump(results, f)

    return results




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


def load_model_and_tokenizer():
    model_name = os.environ["MODEL"]
    original_model_name = os.environ["ORIGINAL_MODEL"]
    tokenizer = AutoTokenizer.from_pretrained(original_model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_dataset_from_file(dataset_path: str, file_format: FileFormat):
    if file_format == FileFormat.CSV:
        return load_dataset("csv", data_files=dataset_path)
    elif file_format == FileFormat.JSON:
        return load_dataset("json", data_files=dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

def prepare_dataset(dataset, tokenizer, dataset_type: DatasetType | CustomDatasetType):
    def tokenize_function(examples):
        if isinstance(dataset_type, DatasetType):
            if dataset_type == DatasetType.INSTRUCT:
                text = [f"Instruction: {instr}\nInput: {inp}\nOutput: {out}" 
                        for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"])]
            elif dataset_type == DatasetType.PRETRAIN:
                text = examples["input"]
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
        elif isinstance(dataset_type, CustomDatasetType):
            text = [dataset_type.format.format(**example) for example in examples]
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        return tokenizer(text, truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def _calculate_evaluation_metrics(total_loss: float, num_batches: int) -> dict[str, float]:
    """Calculate evaluation metrics based on total loss and number of batches."""
    if num_batches > 0:
        average_loss = total_loss / num_batches
        logger.info(f"Average loss: {average_loss}")
    else:
        logger.error("No valid batches were processed during evaluation.")
        average_loss = float("inf")

    return {
        "average_loss": average_loss,
        "perplexity": 2 ** average_loss,
    }

def run_evaluation():
    dataset = os.environ.get("DATASET")
    model = os.environ.get("MODEL")
    original_model = os.environ.get("ORIGINAL_MODEL")
    dataset_type = os.environ.get("DATASET_TYPE")
    file_format = os.environ.get("FILE_FORMAT")

    model, tokenizer = load_model_and_tokenizer(model)

    if file_format == FileFormat.HF:
        raw_dataset = load_dataset(dataset)
    else:
        raw_dataset = load_dataset_from_file(dataset, FileFormat(file_format))

    if dataset_type == "custom":
        dataset_type = CustomDatasetType(**json.loads(os.environ.get("DATASET_TYPE_CONFIG", "{}")))
    else:
        dataset_type = DatasetType(dataset_type)
    
    tokenized_dataset = prepare_dataset(raw_dataset, tokenizer, dataset_type)

    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tokenized_dataset["train"]:
            inputs = {k: torch.tensor(v).unsqueeze(0) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            num_batches += 1

    results = _calculate_evaluation_metrics(total_loss, num_batches)
    
    is_finetune = model_is_a_finetune(original_model, model)
    
    results["is_finetune"] = is_finetune

    with open('/app/results/evaluation_results.json', 'w') as f:
        json.dump(results, f)

    logger.info(f"Evaluation results: {results}")
