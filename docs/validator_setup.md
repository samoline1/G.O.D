# Validator Setup Guide

Steps to set-up the validator.

## Prerequisites

- Docker
- Python 3.8+
- Hugging Face account, LLM API token

## Setup Steps

1. Install system dependencies (Ubuntu 24.04 LTS):

```bash
sudo apt update && sudo apt install snapd python3.10 python3.10-venv
python3.10 -m ensurepip --upgrade
sudo snap install task --classic
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2. Create and set up the `.vali.env` file:

You'll need to setup a MINIO account and add in your access key, endpoint and security key below and also a huggingface token in order to access gated models (such as Llama3.2)

To generate the synthetic data you'll also need to either setup your own LLM server following the instructions (TBA later) or use CorcelAPI and add
your corcel token below:

```bash
python3 -m core.create_config
```

Link to fiber

```base
fiber-post-ip --netuid 176 --subtensor.network test --external_port 9001 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```
3. Run service dependencies for local development:

```bash
task dev_setup
```

4. Run the Validator service on host with refresh for development:

```bash
task validator
```
