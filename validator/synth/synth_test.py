import asyncio
from validator.synth.synth import generate_synthetic_dataset
from fiber.logging_utils import get_logger
logger = get_logger(__name__)

async def main():
    dataset_name = "stanfordnlp/imdb"
    
    synthetic_dataset = await generate_synthetic_dataset(dataset_name)
    
    logger.info(f"Number of synthetic samples generated: {len(synthetic_dataset)}")
    
    logger.info("Synthetic Dataset Samples:")
    for i in range(5):
        logger.info(f"Sample {i+1}:")
        logger.info(synthetic_dataset[i])

if __name__ == "__main__":
    asyncio.run(main())
