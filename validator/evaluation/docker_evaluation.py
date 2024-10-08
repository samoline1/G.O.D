import docker
import json
import io
import tarfile
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

        tar_stream, _ = container.get_archive(cst.CONTAINER_EVAL_RESULTS_PATH)

        file_like_object = io.BytesIO()
        for chunk in tar_stream:
            file_like_object.write(chunk)
        file_like_object.seek(0)

        # LLM Wrote me, I work, apparently optimised for speed, note to come back to
        with tarfile.open(fileobj=file_like_object) as tar:
            # List the members of the tar file
            members = tar.getnames()
            logger.debug(f"Tar archive members: {members}")

            eval_results_file = None
            for member in members:
                if member.endswith('evaluation_results.json'):
                    eval_results_file = tar.extractfile(member)
                    break

            if eval_results_file is None:
                raise Exception("Evaluation results file not found in tar archive")

            eval_results_content = eval_results_file.read().decode('utf-8')
            eval_results = json.loads(eval_results_content)

        container.remove()
        return EvaluationResult(**eval_results)
    except Exception as e:
        logger.error(f"Failed to retrieve evaluation results: {str(e)}")
        raise Exception(f"Failed to retrieve evaluation results: {str(e)}")
    finally:
        client.close()

