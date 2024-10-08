import asyncio
from validator.evaluation.task_prep import prepare_task
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

async def main():
    dataset_name = "mhenrichsen/alpaca_2k_test"
    columns_to_sample = ['input', 'output', 'instruction', 'text']
    repo_name = "cwaud/test_ds"
    
    combined_test_dataset = await prepare_task(dataset_name, columns_to_sample, repo_name)
    
    logger.info(f"Combined test dataset size: {len(combined_test_dataset)}")
    # show first 5 samples
    logger.info(f"Combined test dataset samples: {combined_test_dataset[:5]}")


if __name__ == "__main__":
    asyncio.run(main())