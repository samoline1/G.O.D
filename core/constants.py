import os
from dotenv import load_dotenv

load_dotenv()

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"

CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'
VALI_CONFIG_PATH = "validator/test_axolotl.yml"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")

# Task Stuff
MINIMUM_MINER_POOL = 1 # we need at least 4 miners
REPO_OWNER = 'cwaud' # please change from mine :D

# scoring stuff
MAX_COMPETITION_HOURS = 10
SOFTMAX_TEMPERATURE = 0.5
TEST_SCORE_WEIGHTING = 0.8 # synth will be (1 - this)
TARGET_SCORE_RATIO = 2
