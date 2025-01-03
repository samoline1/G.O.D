import base64
import json
import os
from typing import Dict
from typing import List
from typing import Tuple

from fiber.logging_utils import get_logger
from validator.core.models import Img2ImgPayload

from validator.evaluation.utils import base64_to_image
from validator.evaluation.utils import calculate_l2_loss
from validator.evaluation.utils import download_from_huggingface
from validator.core import constants as cst
from validator.utils import comfy_api_gate as api_gate


logger = get_logger(__name__)


def load_comfy_workflows():
    with open(cst.LORA_WORKFLOW_PATH, "r") as file:
        lora_template = json.load(file)

    return lora_template


def edit_workflow(payload: Dict, edit_elements: Img2ImgPayload, text_guided: bool) -> Dict:
    payload["Checkpoint_loader"]["inputs"]["ckpt_name"] = edit_elements.ckpt_name
    payload["Sampler"]["inputs"]["steps"] = edit_elements.steps
    payload["Sampler"]["inputs"]["cfg"] = edit_elements.cfg
    payload["Sampler"]["inputs"]["denoise"] = edit_elements.denoise
    payload["Image_loader"]["inputs"]["image"] = edit_elements.base_image
    payload["Lora_loader"]["inputs"]["lora_name"] = edit_elements.lora_name
    if text_guided:
        payload["Prompt"]["inputs"]["text"] = edit_elements.prompt
    else:
        payload["Prompt"]["inputs"]["text"] = ""

    return payload


def inference(image_base64: str, params: Dict, use_prompt: bool = False, prompt: str = None) -> Tuple[float, float]:
    if use_prompt and prompt:
        params.prompt = prompt

    params.base_image = image_base64

    lora_payload = edit_workflow(params.comfy_template, params, text_guided=use_prompt)
    lora_gen = api_gate.generate(lora_payload)[0]
    lora_gen_loss = calculate_l2_loss(base64_to_image(image_base64), lora_gen)

    return lora_gen_loss


def eval_loop(dataset_path: str, params: Img2ImgPayload) -> Dict[str, List]:
    lora_losses_text_guided = []
    lora_losses_no_text = []

    for file_name in os.listdir(dataset_path):
        if file_name.lower().endswith(cst.SUPPORTED_FILE_EXTENSIONS):
            logger.info(f"Calculating losses for {file_name}")
            base_name = os.path.splitext(file_name)[0]
            png_path = os.path.join(dataset_path, file_name)
            txt_path = os.path.join(dataset_path, f"{base_name}.txt")

            with open(png_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            prompt = None
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as text_file:
                    prompt = text_file.read()

            lora_gen_loss_text = inference(image_base64, params, use_prompt=True, prompt=prompt)

            logger.info(f"Text guided loss for {file_name} on lora model: {lora_gen_loss_text}")

            lora_gen_loss_no_text = inference(image_base64, params, use_prompt=False)

            logger.info(f"No text loss for {file_name} on lora model: {lora_gen_loss_no_text}")

            lora_losses_text_guided.append(lora_gen_loss_text)
            lora_losses_no_text.append(lora_gen_loss_no_text)

    return {"text_guided_losses": lora_losses_text_guided, "no_text_losses": lora_losses_no_text}


def main():
    test_dataset_path = os.getenv("TEST_DATASET_PATH")
    trained_lora_model_repos = os.getenv("TRAINED_LORA_MODEL_REPOS").split(",")
    base_model_repo = os.getenv("BASE_MODEL_REPO")
    base_model_filename = os.getenv("BASE_MODEL_FILENAME")
    lora_model_filenames = os.getenv("LORA_MODEL_FILENAMES").split(",")

    # Base model download
    logger.info("Downloading base model")
    _ = download_from_huggingface(base_model_repo, base_model_filename, cst.CHECKPOINTS_SAVE_PATH)
    logger.info("Base model downloaded")

    loras_to_evaluate = {}

    for lora_repo, lora_filename in zip(trained_lora_model_repos, lora_model_filenames):
        lora_metadata = {"hf_filename": lora_filename, "hf_repo": lora_repo}
        lora_metadata["local_model_path"] = download_from_huggingface(lora_repo, lora_filename, cst.LORAS_SAVE_PATH)
        loras_to_evaluate[f"{lora_repo}/{lora_filename}"] = lora_metadata

    lora_comfy_template = load_comfy_workflows()
    api_gate.connect()

    results = {}

    for lora_key, lora_metadata in loras_to_evaluate.items():
        img2img_payload = Img2ImgPayload(
            ckpt_name=base_model_filename,
            lora_name=os.path.basename(lora_metadata["hf_filename"]),
            steps=cst.DEFAULT_STEPS,
            cfg=cst.DEFAULT_CFG,
            denoise=cst.DEFAULT_DENOISE,
            comfy_template=lora_comfy_template,
        )

        loss_data = eval_loop(test_dataset_path, img2img_payload)

        repo = lora_metadata["hf_repo"]
        filename = lora_metadata["hf_filename"]

        results[f"{repo}/{filename}"] = loss_data
        if os.path.exists(lora_metadata["local_model_path"]):
            os.remove(lora_metadata["local_model_path"])

    eval_losses = {"eval_losses": results}
    output_file = "/aplp/evaluation_results.json"
    output_dir = os.path.dirname(output_file)

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the results to the file
    with open(output_file, "w") as f:
        json.dump(eval_losses, f)

    logger.info(f"Evaluation results saved to {output_file}")

    logger.info(json.dumps(results))


if __name__ == "__main__":
    main()
