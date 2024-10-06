#!/bin/bash

git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl

sed -i 's/xformers==0.0.27/xformers==0.0.28/' requirements.txt

pip install -e ".[flash-attn,deepspeed]"

cd ..
rm -rf axolotl