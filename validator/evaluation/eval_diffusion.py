import base64
import copy
import json
import os
import subprocess
from io import BytesIO
from typing import Dict

import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image


lora_workflow_path = "comfy_workflows/lora.json"
with open(lora_workflow_path, "r") as file:
    lora_template = json.load(file)

base_workflow_path = "comfy_workflows/base.json"
with open(base_workflow_path, "r") as file:
    base_template = json.load(file)


def start_comfyui():
    try:
        subprocess.run(["./start_comfy.sh", "start"], check=True)
        print("ComfyUI started.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start ComfyUI: {e}")


def stop_comfyui():
    try:
        subprocess.run(["./start_comfy.sh", "stop"], check=True)
        print("ComfyUI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop ComfyUI: {e}")


def base64_to_image(base64_string: str) -> Image.Image:
    image_data = base64.b64decode(base64_string)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image


def download_from_huggingface(repo_id, filename=None, local_dir=None):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"File downloaded to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


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


def inference(
    image_base64: str, base_workflow: Dict, lora_workflow: Dict, params: Dict, use_prompt: bool = False, prompt: str = None
):
    import comfy_api_gate as api_gate

    params_payload = copy.deepcopy(params)
    if use_prompt and prompt:
        params_payload["prompt"] = prompt

    params_payload["image_b64"] = image_base64

    base_payload = edit_workflow(base_workflow, False, params_payload, text_guided=use_prompt)
    base_gen = api_gate.generate(base_payload)[0]
    base_gen_loss = calculate_l2_loss(base64_to_image(image_base64), base_gen)

    lora_payload = edit_workflow(lora_workflow, True, params_payload, text_guided=use_prompt)
    lora_gen = api_gate.generate(lora_payload)[0]
    lora_gen_loss = calculate_l2_loss(base64_to_image(image_base64), lora_gen)

    return base_gen_loss, lora_gen_loss


def eval_loop(dataset_path: str, params: Dict):
    loss_differences_text_guided = []
    loss_differences_no_text = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith(".png"):
            base_name = os.path.splitext(file_name)[0]
            png_path = os.path.join(dataset_path, file_name)
            txt_path = os.path.join(dataset_path, f"{base_name}.txt")

            with open(png_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

            prompt = None
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as text_file:
                    prompt = text_file.read()

            base_gen_loss_text, lora_gen_loss_text = inference(
                image_base64, base_template, lora_template, params, use_prompt=True, prompt=prompt
            )
            base_gen_loss_no_text, lora_gen_loss_no_text = inference(
                image_base64, base_template, lora_template, params, use_prompt=False
            )

            loss_differences_text_guided.append(float(base_gen_loss_text - lora_gen_loss_text))
            loss_differences_no_text.append(float(base_gen_loss_no_text - lora_gen_loss_no_text))

    return loss_differences_text_guided, loss_differences_no_text


def main():
    start_comfyui()
    test_dataset_path = "/root/anime-test-split"
    trained_lora_model_repo = "diagonalge/animelora"
    base_model_repo = "SG161222/RealVisXL_V4.0"
    base_model_filename = "RealVisXL_V4.0.safetensors"
    lora_model_filename = "test/kohyatest.safetensors"
    download_from_huggingface(base_model_repo, base_model_filename, "ComfyUI/models/checkpoints")
    download_from_huggingface(trained_lora_model_repo, lora_model_filename, "ComfyUI/models/loras")

    params = {"ckpt_name": base_model_filename, "lora_name": "kohyatest.safetensors", "steps": 10, "cfg": 4, "denoise": 0.9}

    loss_differences_text_guided, loss_differences_no_text = eval_loop(test_dataset_path, params)
    print(f"Text guided differences: {loss_differences_text_guided}")
    print(f"No Text difference: {loss_differences_no_text}")
    print(f"Cumulative loss difference sum: {sum(loss_differences_text_guided) + sum(loss_differences_no_text)}")

    stop_comfyui()


if __name__ == "__main__":
    main()
