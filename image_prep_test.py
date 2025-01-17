import asyncio
from validator.tasks.task_prep import prepare_image_task

async def main():
    result = await prepare_image_task("/root/Dataset.zip")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
