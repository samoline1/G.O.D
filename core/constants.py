import os
from dotenv import load_dotenv

load_dotenv()

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'
VALI_CONFIG_PATH = "validator/test_axolotl.yml"  
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"

CONTAINER_EVAL_RESULTS_PATH = "/app/evaluation_results.json"
PERCENTAGE_SYNTH = 0.1 
