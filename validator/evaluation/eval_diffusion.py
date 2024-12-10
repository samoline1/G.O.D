import subprocess

from huggingface_hub import hf_hub_download


def start_comfyui():
    """
    Start ComfyUI by calling the bash script.
    """
    try:
        subprocess.run(["./start_comfy.sh", "start"], check=True)
        print("ComfyUI started.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start ComfyUI: {e}")


def stop_comfyui():
    """
    Stop ComfyUI by calling the bash script.
    """
    try:
        subprocess.run(["./start_comfy.sh", "stop"], check=True)
        print("ComfyUI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to stop ComfyUI: {e}")


def download_from_huggingface(repo_id, filename=None, local_dir=None):
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        print(f"File downloaded to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


def main():
    # test_dataset_path = os.environ.get("DATASET")
    trained_lora_model = "diagonalge/animelora"
    base_model = "SG161222/RealVisXL_V4.0"
    download_from_huggingface(
        base_model, "RealVisXL_V4.0.safetensors", "/root/tuning/validator/evaluation/ComfyUI/models/checkpoints"
    )
    download_from_huggingface(
        trained_lora_model, "test/kohyatest.safetensors", "/root/tuning/validator/evaluation/ComfyUI/models/loras"
    )
    start_comfyui()
    stop_comfyui()


if __name__ == "__main__":
    main()
