# Miner Setup Guide

Setting up a miner.

## Prerequisites

- Docker
- Python 3.8+
- Hugging Face account and API token
- WANDB token for training

## Setup Steps

1. Install system dependencies:

```bash
sudo apt update && sudo apt install snapd python3.12 python3.12-venv
python3.12 -m ensurepip --upgrade
sudo snap install task --classic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2. Set up training environment variables in `.env`:

   - `HUGGINGFACE_TOKEN`
   - `WANDB_TOKEN`

3. Start the miner service:

```bash
task miner
```

# Things to consider

The base code accepts all jobs and runs them all with a single config found in [configs/base.yml](configs/base.yml) - you might not want to run the
same traning code with every job. The base code also will write to the repo found in the config. You will probably want to make this
dynamic else you will be overwritting the same model over and over.

As jobs come in you will only have a short amount of time to complete the training, you will first recieve a request into your task_offer endpoint with a bit
of details about the training task. If you reply with accept you will have signed up to the job and will be score accordingly - whether you complete the training and
return a model or not (null returns will result in 0 score).

Once a job is accepted, it will take a short amount of time while other contenders are found and the data is prepared (< 15 minutes) you will then recieve a 'start_training'
post request with the full data details and your training script will automatically begin.

After the alloted time for a competition, the validator will request your submission through the get_latest_submission endpoint which needs to be a huggingface repo containing a fine-tune (lora is fine)
of the model you have been allocated. [endpoint details found here](miner/endpoints/tuning.py).

