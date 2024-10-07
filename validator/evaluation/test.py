from validator.evaluation.docker_evaluation import run_evaluation_docker  
import os
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    results = run_evaluation_docker(
        dataset="mhenrichsen/alpaca_2k_test",
        model="unsloth/Llama-3.2-3B-Instruct",
        original_model="unsloth/Llama-3.2-3B-Instruct",
        dataset_type="custom",
        file_format="hf",
        system_prompt="you are helpful",
        system_format="{system}",
        field_system="text",
        field_instruction="instruction",
        field_input="input",
        field_output="output"
    )
    logger.info(f"Evaluation results: {results}")