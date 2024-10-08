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



PROMPT_GEN_ENDPOINT = "https://api.corcel.io/v1/chat/completions"
PROMPT_GEN_TOKEN = os.getenv("CORCEL_TOKEN")
PROMPT_PATH = "validator/prompts.yaml"
PERCENTAGE_SYNTH = 0.1 
