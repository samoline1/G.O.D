import os
from dotenv import load_dotenv
from fiber.logging_utils import get_logger

logger = get_logger(__name__)
load_dotenv()

DOCKER_IMAGE = "weightswandering/tuning_miner:latest"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'
VALI_CONFIG_PATH = "validator/test_axolotl.yml"  
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
logger.debug(f"HUGGINGFACE_TOKEN: {HUGGINGFACE_TOKEN}")

