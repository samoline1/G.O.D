import asyncio
from validator.synth.synth import generate_synthetic_dataset, load_and_sample_dataset
from fiber.logging_utils import get_logger

logger = get_logger(__name__)

async def main():
    dataset_name = "mhenrichsen/alpaca_2k_test"
    columns_to_sample = ["instruction", "input", "output"]
    
    sampled_data = load_and_sample_dataset(dataset_name, columns_to_sample)
    synthetic_dataset = await generate_synthetic_dataset(sampled_data)
    
    logger.info(f"Number of synthetic samples generated: {len(synthetic_dataset)}")

if __name__ == "__main__":
    asyncio.run(main())
