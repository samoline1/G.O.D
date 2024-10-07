import docker
from typing import Union
from core.models.payload_models import DatasetType, CustomDatasetType, FileFormat, EvaluationResult
from core import constants as cst
from fiber.logging_utils import get_logger
import json

logger = get_logger(__name__)

def run_evaluation_docker(
    dataset: str,
    model: str,
    original_model: str,
    dataset_type: Union[DatasetType, CustomDatasetType],
    file_format: FileFormat
) -> EvaluationResult:
    client = docker.from_env()

    environment = {
        "DATASET": dataset,
        "MODEL": model,
        "ORIGINAL_MODEL": original_model,
        "FILE_FORMAT": file_format.value,
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
    }

    if isinstance(dataset_type, CustomDatasetType):
        environment["DATASET_TYPE"] = dataset_type.model_dump_json()
    else:
        environment["DATASET_TYPE"] = dataset_type.value

    try:
        container = client.containers.run(
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(
                count=-1, capabilities=[['gpu']]
            )],
            detach=True,
        )

        # Stream logs in real-time
        for log in container.logs(stream=True, follow=True):
            logger.info(f"Container log: {log.decode('utf-8').strip()}")

        result = container.wait()

        if result["StatusCode"] != 0:
            logs = container.logs().decode("utf-8")
            raise Exception(
                f"Evaluation failed with status code {result['StatusCode']}: {logs}"
            )

        # Retrieve evaluation results from container logs
        logs = container.logs().decode("utf-8")
        try:
            # Assuming eval.py prints the JSON result to stdout
            eval_results = json.loads(logs.strip())
        except json.JSONDecodeError:
            raise Exception("Failed to parse evaluation results from container logs")

        container.remove()
        return EvaluationResult(**eval_results)
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    finally:
        client.close()

