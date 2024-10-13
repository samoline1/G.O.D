import os
from dotenv import load_dotenv

load_dotenv()

MINER_DOCKER_IMAGE = "weightswandering/tuning_miner:latest"
VALIDATOR_DOCKER_IMAGE = "weightswandering/tuning_vali:latest"

CONFIG_DIR = "./core/config/"
OUTPUT_DIR = "./core/outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'
VALI_CONFIG_PATH = "validator/test_axolotl.yml"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
WANDB_TOKEN = os.getenv("WANDB_TOKEN")


# scoring stuff
MAX_COMPETITION_HOURS = 10
TOP_SYNTH_PERCENT_CUTOFF = 0.75 # if your loss is in the top % then you will get a reduced score
PENALISATION_FACTOR_FOR_HIGH_SYNTH_LOSS = 0.5
TEST_SCORE_WEIGHTING = 0.8 # synth will be (1 - this)
