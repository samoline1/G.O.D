import asyncio
from validator.synth.synth import generate_synthetic_dataset

async def main():
    dataset_name = "mhenrichsen/alpaca_2k_test"
    
    synthetic_dataset = await generate_synthetic_dataset(dataset_name)
    
    print(f"Number of synthetic samples generated: {len(synthetic_dataset)}")
    
    print("Synthetic Dataset Samples:")
    for i in range(5):
        print(f"Sample {i+1}:")
        print(synthetic_dataset[i])
        print()

if __name__ == "__main__":
    asyncio.run(main())
