import os
from dotenv import load_dotenv
load_dotenv()

DOCKER_IMAGE = "winglian/axolotl-cloud:main-latest"

CONFIG_DIR = "./configs/"
OUTPUT_DIR = "./outputs/"

CONFIG_TEMPLATE_PATH = CONFIG_DIR + 'base.yml'
VALI_CONFIG_PATH = "validator/test_axolotl.yml"  
PROMPT_PATH = "validator/prompts.yml"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

PERCENTAGE_SYNTH = 0.1  # 10% of the dataset will be used for synthetic data creation

PROMPT_GEN_ENDPOINT = "corcelapi" 
PROMPT_GEN_TOKEN = os.getenv("CORCEL_TOKEN")