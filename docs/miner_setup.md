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

Make sure you have bittensor installed to the latest version then:

```bash
btcli wallet new-coldkey
btcli wallet new-hotkey
```

To check see your wallet address'

```bash
btcli wallet list
```

Once you have some key on your coldkey. Register to the subnet.

```bash
btcli s register --network test
```

Then to connect to fiber

```
fiber-post-ip --netuid 176 --subtensor.network test --external_port 7999 --wallet.name default --wallet.hotkey default --external_ip [YOUR-IP]
```

4. Configure environment variables:
    Create a `.1.env` file with the following (you'll need a huggingface and wandb token):

```
python3 -m core.create_config --miner

```

5. Update the configuration:
    - Update your `wandb_entity` in the wandb section of the config to be your wandb username+org_name [here](../core/config/base.yml)
    - In the same config, change the `hub_repo' to be your huggingface repo that you want the saves (under job id) to be autosaved to

6. Start the miner service:
    ```bash
    task miner
    ```
