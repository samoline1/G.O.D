import os


SUCCESS = "success"
ACCOUNT_ID = "account_id"
MESSAGE = "message"
AMOUNT = "amount"
UNDELEGATION = "undelegation"
STAKE = "stake"
VERIFIED = "verified"
REDIS_KEY_COLDKEY_STAKE = "coldkey_stake"
API_KEY = "api_key"
COLDKEY = "coldkey"


VALI_CONFIG_PATH = "validator/test_axolotl.yml"

# data stuff
TEST_SIZE = 0.1
TRAIN_TEST_SPLIT_PERCENTAGE = 0.1
GET_SYNTH_DATA = True
MAX_SYNTH_DATA_POINTS = 1000
ADDITIONAL_SYNTH_DATA_PERCENTAGE = 0.1

# synth stuff
SYNTH_GEN_BATCH_SIZE = 3
SYNTH_MODEL_TEMPERATURE = 0.3
SYNTH_MODEL = "llama-3-1-8b"
CONTAINER_EVAL_RESULTS_PATH = "/aplp/evaluation_results.json"
PROMPT_GEN_ENDPOINT = "https://api.corcel.io/v1/chat/completions"
PROMPT_GEN_TOKEN = os.getenv("CORCEL_TOKEN")
PROMPT_PATH = "validator/prompts.yml"

# Task Stuff
MINIMUM_MINER_POOL = 1  # we need at least 4 miners usually - this is for testing

# scoring stuff
MAX_COMPETITION_HOURS = 10
SOFTMAX_TEMPERATURE = 0.5
TEST_SCORE_WEIGHTING = 0.8  # synth will be (1 - this)
TARGET_SCORE_RATIO = 2

# processing stuff
MAX_CONCURRENT_MINER_ASSIGNMENTS = 5
MAX_CONCURRENT_TASK_PREPS = 3
MAX_CONCURRENT_TRAININGS = 10
MAX_CONCURRENT_EVALUATIONS = 1
