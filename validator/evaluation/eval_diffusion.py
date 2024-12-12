import base64
import copy
import json
import os
import subprocess
import tempfile
from io import BytesIO
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from fiber.logging_utils import get_logger
from huggingface_hub import hf_hub_download
from PIL import Image

import core.constants as cst

from ..validator.utils import comfy_api_gate as api_gate


logger = get_logger(__name__)


def load_comfy_workflows():
    with open(cst.LORA_WORKFLOW_PATH, "r") as file:
        lora_template = json.load(file)

    with open(cst.BASE_WORKFLOW_PATH, "r") as file:
        base_template = json.load(file)

    return base_template, lora_template


def start_comfyui():
    try:
        subprocess.run(["./start_comfy.sh", "start"], check=True)
        logger.info("ComfyUI started")
    except subprocess.CalledProcessError as e:
        logger.info(f"Failed to start ComfyUI: {e}")


def stop_comfyui():
    try:
        subprocess.run(["./start_comfy.sh", "stop"], check=True)
        logger.info("ComfyUI stopped.")
    except subprocess.CalledProcessError as e:
        logger.info(f"Failed to stop ComfyUI: {e}")


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id: str, filename: str, local_dir: str = None):
    # Need to use a temp folder to make sure we place the files correctly even if its inside a folder on HF
    try:
        local_filename = os.path.basename(filename)
        final_path = os.path.join(local_dir, local_filename)
        if os.path.exists(final_path):
            logger.info(f"File {filename} already exists. Skipping download.")
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=temp_dir)
                os.rename(temp_file_path, final_path)
            logger.info(f"File {filename} downloaded successfully")
    except Exception as e:
        logger.info(f"Error downloading file: {e}")


def calculate_l2_loss(image1: Image.Image, image2: Image.Image) -> float:
    image1 = np.array(image1.convert("RGB")) / 255.0
    image2 = np.array(image2.convert("RGB")) / 255.0
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions to calculate L2 loss.")
    l2_loss = np.mean((image1 - image2) ** 2)
    return l2_loss


def edit_workflow(payload: Dict, lora_workflow: bool, edit_elements: Dict, text_guided: bool) -> Dict:
    payload["Checkpoint_loader"]["inputs"]["ckpt_name"] = edit_elements["ckpt_name"]
    payload["Sampler"]["inputs"]["steps"] = edit_elements["steps"]
    payload["Sampler"]["inputs"]["cfg"] = edit_elements["cfg"]
    payload["Sampler"]["inputs"]["denoise"] = edit_elements["denoise"]
    payload["Image_loader"]["inputs"]["image"] = edit_elements["image_b64"]
    if text_guided:
        payload["Prompt"]["inputs"]["text"] = edit_elements["prompt"]
    else:
        payload["Prompt"]["inputs"]["text"] = ""
    if lora_workflow:
        payload["Lora_loader"]["inputs"]["lora_name"] = edit_elements["lora_name"]
    return payload


def inference(image_base64: str, params: Dict, use_prompt: bool = False, prompt: str = None) -> Tuple[float, float]:
    params_payload = copy.deepcopy(params)
    if use_prompt and prompt:
        params_payload["prompt"] = prompt

    params_payload["image_b64"] = image_base64

    base_payload = edit_workflow(params["base_template"], False, params_payload, text_guided=use_prompt)
    base_gen = api_gate.generate(base_payload)[0]
    base_gen_loss = calculate_l2_loss(base64_to_image(image_base64), base_gen)

    lora_payload = edit_workflow(params["lora_template"], True, params_payload, text_guided=use_prompt)
    lora_gen = api_gate.generate(lora_payload)[0]
    lora_gen_loss = calculate_l2_loss(base64_to_image(image_base64), lora_gen)

    return base_gen_loss, lora_gen_loss


def eval_loop(dataset_path: str, params: Dict) -> Dict[str, List]:
    loss_differences_text_guided = []
    loss_differences_no_text = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".png"):
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

            base_gen_loss_text, lora_gen_loss_text = inference(image_base64, params, use_prompt=True, prompt=prompt)

            logger.info(f"Text guided loss for {file_name} on base model: {base_gen_loss_text}")
            logger.info(f"Text guided loss for {file_name} on lora model: {lora_gen_loss_text}")

            base_gen_loss_no_text, lora_gen_loss_no_text = inference(image_base64, params, use_prompt=False)

            logger.info(f"No text loss for {file_name} on base model: {base_gen_loss_no_text}")
            logger.info(f"No text loss for {file_name} on lora model: {lora_gen_loss_no_text}")

            loss_differences_text_guided.append(float(base_gen_loss_text - lora_gen_loss_text))
            loss_differences_no_text.append(float(base_gen_loss_no_text - lora_gen_loss_no_text))

    return {"text_guided_loss_differences": loss_differences_text_guided, "no_text_loss_differences": loss_differences_no_text}


def main():
    start_comfyui()
    test_dataset_path = "/root/anime-test-split"
    trained_lora_model_repo = "diagonalge/animelora"
    base_model_repo = "SG161222/RealVisXL_V4.0"
    base_model_filename = "RealVisXL_V4.0.safetensors"
    lora_model_filename = "test/kohyatest.safetensors"

    download_from_huggingface(base_model_repo, base_model_filename, "ComfyUI/models/checkpoints")
    download_from_huggingface(trained_lora_model_repo, lora_model_filename, "ComfyUI/models/loras")

    base_comfy_template, lora_comfy_template = load_comfy_workflows()
    params = {
        "ckpt_name": base_model_filename,
        "lora_name": "kohyatest.safetensors",
        "steps": 10,
        "cfg": 4,
        "denoise": 0.9,
        "base_template": base_comfy_template,
        "lora_template": lora_comfy_template,
    }

    api_gate.connect()

    results = eval_loop(test_dataset_path, params)
    results["cumulative_loss_difference"] = sum(results["text_guided_loss_differences"]) + sum(
        results["no_text_loss_differences"]
    )

    output_file = "/aplp/evaluation_results_diffusion.json"
    output_dir = os.path.dirname(output_file)

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the results to the file
    with open(output_file, "w") as f:
        json.dump(results, f)

    logger.info(f"Evaluation results saved to {output_file}")

    logger.info(json.dumps(results))

    stop_comfyui()


if __name__ == "__main__":
    main()
