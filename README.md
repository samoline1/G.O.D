# Tuning Subnet

🚀 Welcome to the Tuning Subnet:

We're just warming up - MVPin'


## Setup

### Prerequisites

- Docker
- Python 3.8+
- Hugging Face account, API token and corcel token (for synth gen, validators only), WANDB token for training

### Miner Setup


1) Install system dependencies

```
sudo apt update && sudo apt install snapd python3.12 python3.12-venv
python3.12 -m ensurepip --upgrade
sudo snap install task --classic
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

2) set-up training environment variables

export HUGGINGFACE_TOKEN
export WANDB_TOKEN


3) Start the miner service


```
task miner
```


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

# Why do I need to export  hugging face token and wandb token then - can't i add them to a .env file?
# We should just add these to the create configs
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
