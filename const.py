import os
from dotenv import load_dotenv
load_dotenv()

DOCKER_IMAGE = "winglian/axolotl:main-latest"

CONFIG_DIR = "./configs/"
OUTPUT_DIR = "./outputs"
COMPLETED_MODEL_DIR = "completed-model"
CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "Tuning")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "rayonlabs-rayon-labs")

# Used for huggingface uploads
REPO = 'tau-vision'
# need to update this with the id? 
USR = 'test_user'

