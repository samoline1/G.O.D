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

You'll need to setup a [min.io](https://min.io/) account (s3 bucket to store the data for miners) and add in your access key, endpoint and security key below.

To generate the synthetic data you'll also need to either setup your own LLM server and give this to the create_config as the gpu_ip or you can leave this blank and instead use chutes-api, you can grab a token from use and add
your token when prompted for 'api_token' when running the next step:

```bash
python3 -m core.create_config
```

Link to fiber

```base
fiber-post-ip --netuid [NET_ID] --subtensor.network test --external_port 9001 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```
3. Run service dependencies for local development:

```bash
task dev_setup
```

4. Run the Validator service on host with refresh for development:

```bash
task validator
```

