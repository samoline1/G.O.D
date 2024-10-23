# Validator Setup Guide

Steps to set-up the validator.

## Prerequisites

- Docker
- Python 3.8+
- Hugging Face account, LLM API token

## Setup Steps

1. Install system dependencies (Ubuntu 24.04 LTS):

```bash
sudo apt update && sudo apt install snapd python3.12 python3.12-venv
python3.12 -m ensurepip --upgrade
sudo snap install task --classic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2. Create and set up the `.vali.env` file:

You'll need to setup a MINIO account and add in your access key, endpoint and security key below and also a huggingface token in order to access gated models (such as Llama3.2)

To generate the synthetic data you'll also need to either setup your own LLM server following the instructions (TBA later) or use CorcelAPI and add
your corcel token below:

```bash
touch .vali.env
cat << EOF >> .vali.env
POSTGRES_USER=user
POSTGRES_PASSWORD=$(openssl rand -hex 32)
POSTGRES_DB=db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
LOCALHOST=true
REDIS_HOST=localhost
DATABASE_URL=postgres:/user:password@postgresql:5432/db?sslmode=disable
MINIO_ENDPOINT=
MINIO_ACCESS_KEY=
MINIO_SECRET_KEY=
HUGGINGFACE_TOKEN=
CORCEL_TOKEN=
EOF
```

3. Run service dependencies for local development:

```bash
task dev_setup
```

4. Run the Validator service on host with refresh for development:

```bash
task validator
```

Note: The Hugging Face token token should be added to your environment variables or a separate `.env` file.

