import asyncio
import datetime
import os
from concurrent.futures import ThreadPoolExecutor

from minio import Minio

from fiber.logging_utils import get_logger


logger = get_logger(__name__)

# NOTE: wen type hints
# NOTE: POINTS:
# - why are you using a separate event loop
# - why are you using a thread pool executor
# Do we really need to do this object orienteated? Would much
# prefer functional style.
# Add the minio client to config and use it directly
class AsyncMinioClient:
    def __init__(self, endpoint, access_key, secret_key, secure=True, region="us-east-1"):
        self.endpoint = endpoint
        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure, region=region)
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor()

    async def upload_file(self, bucket_name, object_name, file_path):
        func = self.client.fput_object
        args = (bucket_name, object_name, file_path)
        logger.info('Attempting to upload')
        try:
            return await self.loop.run_in_executor(self.executor, func, *args)
        except Exception as e:
            logger.info(f"There was an issue with uploading {e}")

    async def download_file(self, bucket_name, object_name, file_path):
        func = self.client.fget_object
        args = (bucket_name, object_name, file_path)
        logger.info('Attempting to download')
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def delete_file(self, bucket_name, object_name):
        func = self.client.remove_object
        args = (bucket_name, object_name)
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def list_objects(self, bucket_name, prefix=None, recursive=True):
        func = self.client.list_objects
        args = (bucket_name, prefix, recursive)
        logger.info('Listing objects')
        return await self.loop.run_in_executor(self.executor, func, *args)

    async def ensure_bucket_exists(self, bucket_name):
        exists = await self.loop.run_in_executor(self.executor, self.client.bucket_exists, bucket_name)
        if not exists:
            await self.loop.run_in_executor(self.executor, self.client.make_bucket, bucket_name)

    async def get_presigned_url(self, bucket_name, object_name, expires=604800):
        expires_duration = datetime.timedelta(seconds=expires)
        func = self.client.presigned_get_object
        args = (bucket_name, object_name, expires_duration)
        return await self.loop.run_in_executor(self.executor, func, *args)

    def get_public_url(self, bucket_name, object_name):
        return f"https://{self.endpoint}/{bucket_name}/{object_name}"

    def __del__(self):
        self.executor.shutdown(wait=False)


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

async_minio_client = AsyncMinioClient(endpoint=MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY)
