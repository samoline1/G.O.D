import json

import pandas as pd
from loguru import logger

from core.models.utility_models import DatasetType
from core.models.utility_models import FileFormat

import os

from urllib.parse import urlparse
import aiohttp

async def validate_dataset(dataset_path: str, dataset_type: DatasetType, file_format: FileFormat) -> bool:
    try:
        if file_format == FileFormat.CSV:
            df = pd.read_csv(dataset_path)
        elif file_format == FileFormat.JSON:
            logger.info("now is json")
            with open(dataset_path, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        if dataset_type == DatasetType.INSTRUCT:
            required_columns = ["instruction", "input", "output"]
        else:
            required_columns = ["input"]

        return all(col in df.columns for col in required_columns)
    except Exception as e:
        raise ValueError(f"Error validating dataset: {str(e)}")

async def download_s3_file(file_url: str) -> str:
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path)
    local_file_path = os.path.join("/tmp", file_name)

    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            if response.status == 200:
                with open(local_file_path, "wb") as f:
                    f.write(await response.read())
            else:
                raise Exception(f"Failed to download file: {response.status}")

    return local_file_path


