#!/bin/bash
set -ex

env | grep -E 'HUGGINGFACE_TOKEN|WANDB'

echo 'Preparing data...'
pip install mlflow
pip install --upgrade huggingface_hub

if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "Attempting to log in to Hugging Face"
    huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
else
    echo "HUGGINGFACE_TOKEN is not set. Skipping login."
fi

mkdir -p /workspace/axolotl/data/
cp /workspace/input_data/{{DATASET_FILENAME}} /workspace/axolotl/data/{{DATASET_FILENAME}}

echo 'Starting training command'
accelerate launch -m axolotl.cli.train /workspace/axolotl/configs/{{JOB_ID}}.yml