import docker
from typing import Union
from core.models.payload_models import DatasetType, CustomDatasetType, FileFormat, EvaluationResult
import validator.constants as cst
from fiber.logging_utils import get_logger
import json

logger = get_logger(__name__)

def run_evaluation_docker(dataset: str, model: str, original_model: str, dataset_type: Union[DatasetType, CustomDatasetType], file_format: FileFormat) -> EvaluationResult:
    client = docker.from_env()

    environment = {
        "DATASET": dataset,
        "MODEL": model,
        "ORIGINAL_MODEL": original_model,
        "DATASET_TYPE": dataset_type.value if isinstance(dataset_type, DatasetType) else "custom",
        "FILE_FORMAT": file_format.value,
        "HUGGINGFACE_TOKEN": cst.HUGGINGFACE_TOKEN,
    }

    if isinstance(dataset_type, CustomDatasetType):
        environment.update({
            "SYSTEM_PROMPT": dataset_type.system_prompt,
            "SYSTEM_FORMAT": dataset_type.system_format,
            "FIELD_SYSTEM": dataset_type.field_system,
            "FIELD_INSTRUCTION": dataset_type.field_instruction,
            "FIELD_INPUT": dataset_type.field_input,
            "FIELD_OUTPUT": dataset_type.field_output
        })

    try:
        container = client.containers.run(
            cst.VALIDATOR_DOCKER_IMAGE,
            environment=environment,
            runtime="nvidia",
            device_requests=[docker.types.DeviceRequest(count=1, capabilities=[['gpu']])],
            volumes={'/tmp': {'bind': '/app/results', 'mode': 'rw'}},
            detach=True,
        )

        result = container.wait()
        
        if result["StatusCode"] != 0:
            logs = container.logs().decode("utf-8")
            raise Exception(f"Evaluation failed: {logs}")

        _, output = container.get_archive('/app/results/evaluation_results.json')
        file_content = b''.join(output)
        eval_results = json.loads(file_content.decode('utf-8'))

        container.remove()

        return EvaluationResult(**eval_results)

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

    finally:
        client.close()

