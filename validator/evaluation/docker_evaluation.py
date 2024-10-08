import docker
import json
import os
import tarfile
import tempfile
from core.models.payload_models import DatasetType, CustomDatasetType, FileFormat, EvaluationResult
from core import constants as cst
from fiber.logging_utils import get_logger
from core.models.utility_models import DatasetType, CustomDatasetType
from typing import Union

logger = get_logger(__name__)

def run_evaluation_docker(
    dataset: str,
    model: str,
    original_model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat
) -> EvaluationResult:
    client = docker.from_env()

    # Serialize dataset_type for environment variable passing
    if isinstance(dataset_type, DatasetType):
        dataset_type_str = dataset_type.value
    elif isinstance(dataset_type, CustomDatasetType):
        dataset_type_str = dataset_type.model_dump_json()
    else:
        raise ValueError("Invalid dataset_type provided.")

    environment = {
        "DATASET": dataset,
        "MODEL": model,
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type_str,
        "FILE_FORMAT": file_format.value,
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
    }

    try:
        # Run the evaluation container
        container = client.containers.run(
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(
                count=-1, capabilities=[['gpu']]
            )],
            detach=True,
        )

        result = container.wait()

        if result["StatusCode"] != 0:
            logs = container.logs().decode('utf-8')
            logger.error(f"Container exited with status {result['StatusCode']}: {logs}")
            raise Exception(f"Container exited with status {result['StatusCode']}")

        container_results_path = "/app/evaluation_results.json"

        tar_stream, _ = container.get_archive(container_results_path)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tarfile_path = os.path.join(tmpdirname, "evaluation_results.tar")
            with open(tarfile_path, "wb") as f:
                for chunk in tar_stream:
                    f.write(chunk)

            with tarfile.open(tarfile_path) as tar:
                tar.extractall(path=tmpdirname)

            extracted_file_path = os.path.join(tmpdirname, container_results_path.lstrip("/"))
            with open(extracted_file_path, "r") as f:
                eval_results = json.load(f)

        container.remove()
        return EvaluationResult(**eval_results)
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")
    finally:
        client.close()

