import os
from dotenv import load_dotenv
load_dotenv()

DOCKER_IMAGE = "winglian/axolotl-cloud:main-latest"

CONFIG_DIR = "./configs/"
OUTPUT_DIR = "./outputs"
COMPLETED_MODEL_DIR = "completed-model"
CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

print(f"HUGGINGFACE_TOKEN in host environment: {os.getenv('HUGGINGFACE_TOKEN')}")

