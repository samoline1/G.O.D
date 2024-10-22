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
    sudo apt update && sudo apt install snapd python3.10 python3.10-venv
    python3.10 -m ensurepip --upgrade
    sudo snap install task --classic
    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -e '.[dev]'
    pre-commit install
    ```

2. Set up your wallet:
    - Get your wallet (hotkey) on the machine
    - Run `core/create_config.py --miner`
    - For developers: use `SUBTENSOR_ADDRESS=wss://test.finney.opentensor.ai:443/`

3. Configure environment variables:
    Create a `.1.env` file with the following:
    - `HUGGINGFACE_TOKEN`
    - `WANDB_TOKEN`

4. Update the configuration:
    - Update your `entity_id` in the wandb section of the config to be your wandb username+org_name [here](core/config/base.yml)
    - In the same config, change the `hub_model_id` to be the huggingface hub repo you want to upload to

5. Start the miner service:
    ```bash
    task miner
    ```

## Testing

You'll want to check that jobs are accepted as expected. Here's an example payload for testing:

```bash
curl -X POST http://localhost:7999/train/ \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "mhenrichsen/alpaca_2k_test",    # any hf dataset
    "model": "unsloth/Llama-3.2-3B-Instruct",   # any hf model
    "dataset_type": {                           # you can define the columns to use here
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
