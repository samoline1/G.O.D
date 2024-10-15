# Tuning Subnet

Starting point snr

## Setup

### Prerequisites

- Docker
- Python 3.8+
- Hugging Face account, API token and corcel token (for synth gen)

### Miner Setup

Miner is m1_dev task rn, this will start up the server and accept a fine-tuning job.

### Validator Setup

Should be a case of installing requirements and then trying one of the tests - for now.


## Components

### 1. Train Endpoint (Miners Only)

The train endpoint allows you to initiate a fine-tuning job. Takes a dataset, a model, and a dataset type as input and enqueues the job for processing. The job is then processed by a worker in a separate Docker container, the idea is to allow for expansion to kubernetes seamlessly.

#### Endpoint: POST /train/

##### Request Body:
- `dataset`: Path to the dataset file or Hugging Face dataset name
- `model`: Name or path of the model to be trained
- `dataset_type`: Type of the dataset (e.g., "instruct", "pretrain", "alpaca", or a custom one like I shared with you below)
- `file_format`: Format of the dataset file (e.g., "csv", "json", "hf") - csv and json are local formats, hf is a Hugging Face dataset which the container will download.

Example request:

```bash
      curl -X POST http://localhost:7999/train/ \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "mhenrichsen/alpaca_2k_test",
       "model": "unsloth/Llama-3.2-3B-Instruct",
       "dataset_type": {
         "system_prompt": "you are helpful",
         "system_format": "{system}",
         "field_system": "text",
         "field_instruction": "instruction",
         "field_input": "input",
         "field_output": "output"
       },
       "file_format": "hf"
     }'
```

##### Process:
1. Validates the input parameters
2. Creates a job configuration file
3. Enqueues the job for processing
4. Starts a Docker container to run the training process

### 2. Evaluate  (Validators Only)

Best thing to do to test out right now is call one of the three tests set-up.

prepare_task - does dataset divide + synth
test_run_eval - does a full evaluation test and gets back the loss



##### Process for eval:
1. Loads the fine-tuned model and original tokenizer
2. Checks if the provided model is likely a fine-tune of the original model based on model parameters
3. Performs evaluation using a test dataset and gets loss and perplexity

##### Response:
- `is_finetune`: Boolean indicating if the model appears to be a fine-tune
- `eval_results`: Dictionary containing evaluation metrics

## Additional Info

- Miners use a custom `TrainingWorker` class to manage the job queue and process training jobs asynchronously.
- Docker is used by both miners and validators (one each)
- Validators use the Axolotl framework to load and process datasets, and perform custom evaluation loops to calculate loss and perplexity. I needed to write the validation loop to work with the Axolotl framework since the dataset stuff was all handled in the lib, making it easier to work with.
- The hf repo will write to hub_model_id which is inside the configs/base.yml file.



## Validator Local Development installation

1) Install system dependencies (Ubuntu 24.04 LTS)

```
sudo apt update && sudo apt install snapd python3.12 python3.12-venv
python3.12 -m ensurepip --upgrade
sudo snap install task --classic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2) Install .vali.env environment variables:

```
touch .vali.env
cat << EOF >> .vali.env
POSTGRES_USER=user
POSTGRES_PASSWORD=$(openssl rand -hex 32)
POSTGRES_DB=db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
LOCALHOST=true
EOF
```

3) Run Service Dependencies for local development

```
task dev_setup
```

4) Run Validator service on host with refresh for development

```
task validator
```
