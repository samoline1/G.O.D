#!/bin/bash

# Clone the ComfyUI
if [ ! -d ComfyUI ] || [ -z "$(ls -A ComfyUI)" ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
  cd ComfyUI
  git fetch --depth 1 origin a220d11e6b8dd0dbbf50f81ab9398ec202a96fe6
  git checkout a220d11e6b8dd0dbbf50f81ab9398ec202a96fe6
  cd ..
fi

# Create the virtual environment
if [ ! -d "ComfyUI/venv" ]; then
  echo "Creating virtual environment"
  python3 -m venv ComfyUI/venv
fi

cd ComfyUI/custom_nodes

if [ ! -d comfyui-tooling-nodes ]; then
  git clone --depth 1 https://github.com/Acly/comfyui-tooling-nodes
fi

cd ..

venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "Virtual environment created and dependencies installed"

cd ..
