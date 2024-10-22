# Miner Setup Guide

This guide will walk you through the process of setting up and running a miner for the Tuning Subnet.

## Prerequisites

- Docker
- Python 3.8+
- Hugging Face account and API token
- WANDB token for training

## Setup Steps

1. Install system dependencies:

```bash
sudo apt update && sudo apt install snapd python3.l0 python3.10-venv
python3.10 -m ensurepip --upgrade
sudo snap install task --classic
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2. Set up training environment variables in `.1.env`:

   - `HUGGINGFACE_TOKEN`
   - `WANDB_TOKEN`

3.

update your 'entity_id' in the wandb section of the config to be your wandb username+org_name [here](core/config/base.yml)

4. Start the miner service:

```bash
task miner
```


5. Testing


You'll want to check a few things first that jobs are accepted as you expect:


start_traning example payload:

```bash
      curl -X POST http://localhost:7999/train/ \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "mhenrichsen/alpaca_2k_test", # any hf dataset
       "model": "unsloth/Llama-3.2-3B-Instruct", # any hf model
       "dataset_type": {   # you can define the columns to use here
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




