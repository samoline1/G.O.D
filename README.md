# Tuning Subnet

## Training


To run the training worker, you need to have docker installed and running on your machine and export your huggingface token to the environment.

```bash
export HUGGINGFACE_TOKEN=<your_huggingface_token>
python training_worker.py
```

All the training settings are is the configs/base.yml 


## Evaluation

#this stuff needs installling 
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'

Then the requirements for the validation are in the Validation/requirements.txt

