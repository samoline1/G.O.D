# Tuning Subnet

Starting point snr 

## Setup

### Prerequisites

- Docker
- Python 3.8+
- Hugging Face account and API token

### Miner Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/tuning-subnet.git
   cd tuning-subnet
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Hugging Face token:
   ```bash
   export HUGGINGFACE_TOKEN=<your_huggingface_token>
   ```
   # NOTE: need a nicer error if this is not set

I think I covered everything in the requirements - but honestly can't remember - might need a couple more bits, the errors will guide you. 

### Validator Setup

For the validator, we need to set up the Axolotl framework since we're not using the Docker container. 

1. Follow steps 1-3 from the Miner Setup.

2. Install Axolotl dependencies:
   ```bash
   git clone https://github.com/axolotl-ai-cloud/axolotl
   cd axolotl
   pip install packaging ninja
   pip install -e '.[flash-attn,deepspeed]'
   ```

3. Install additional validation requirements:
   ```bash
   pip install -r validator/requirements.txt
   ```

## Running the Service

Not much to it rn 

```bash
python main.py
```

## Components

### 1. Train Endpoint (Miners Only)

The train endpoint allows you to initiate a fine-tuning job. Takes a dataset, a model, and a dataset type as input and enqueues the job for processing. The job is then processed by a worker in a separate Docker container, the idea is to allow for expansion to kubernetes seemlessly. 

#### Endpoint: POST /train/

##### Request Body:
- `dataset`: Path to the dataset file or Hugging Face dataset name
- `model`: Name or path of the model to be trained
- `dataset_type`: Type of the dataset (e.g., "instruct", "pretrain", "alpaca", or a custom one like I shared with you below)
- `file_format`: Format of the dataset file (e.g., "csv", "json", "hf") - csv and json are local formats, hf is a Hugging Face dataset which the container will download. 

Example request:

```bash
      curl -X POST http://localhost:7999/train/ \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "mhenrichsen/alpaca_2k_test",
       "model": "unsloth/Llama-3.2-3B-Instruct",
       "dataset_type": {
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

##### Process:
1. Validates the input parameters (dataset_validation.py)
2. Creates a job configuration file job_handler.py)
3. Enqueues the job for processing (training_worker.py)
4. Starts a Docker container to run the training process (job_handler.py)

### 2. Evaluate Endpoint (Validators Only)

The evaluate endpoint allows you to assess the performance of a fine-tuned model.

#### Endpoint: POST /evaluate/

##### Request Body:
- Same as the train endpoint, plus:
- `original_model`: Name or path of the original model (before fine-tuning)

Example request:


```bash
 curl -X POST http://localhost:8000/evaluate/ \
     -H "Content-Type: application/json" \
     -d '{
       "dataset": "mhenrichsen/alpaca_2k_test",
       "model": "unsloth/Llama-3.2-3B-Instruct",
       "original_model": "unsloth/Llama-3.2-3B-Instruct",
       "dataset_type": {
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

##### Process:
1. Loads the fine-tuned model and original tokenizer (endpoints.py)
2. Checks if the provided model is likely a fine-tune of the original model based on model parameters (checking.py)
3. Performs evaluation using a test dataset and gets loss and perplexity (checking.py)

##### Response:
- `is_finetune`: Boolean indicating if the model appears to be a fine-tune
- `eval_results`: Dictionary containing evaluation metrics

## Additional Info 

- Miners use a custom `TrainingWorker` class to manage the job queue and process training jobs asynchronously.
- Docker is used by miners to isolate the training environment and ensure consistent execution across different systems.
- Validators use the Axolotl framework directly to load and process datasets, and perform custom evaluation loops to calculate loss and perplexity. I needed to write the validation loop to work with the Axolotl framework since the dataset stuff was all handled in the lib, making it easier to work with. 
- The hf repo will write to hub_model_id which is inside the configs/base.yml file. 

### Things for Jefe to Do

- Given a user dataset, we need to split it train/test, and save the train to HF to pass out to miners for tuning. 
- Validators need to select groups of miners to send the same training dataset to, and then we need a scoring mechanism for evaluating the results of the miners given the perplexity scores they return. 
- The passing around of datasets and having miners submit their fine-tuned models to the chain within the allocated time period (https://github.com/macrocosm-os/pretraining/blob/fde1e899f16bc6ad438eee868d26f940d6f8b146/pretrain/mining.py#L56 could be adapted for this)
- Taking my lackluster code and putting it into the format you'd like.
- Probably creating our own Docker container for the miners to use with the added script stuff I needed to add (see scripts/*) - I can help with this bit.

### Things for Both of Us 

- Think about a way of using the ML loss charts to score miners. 
- Create some synthetic data pipeline given the real data so that we can test the miners' models on this data too. If miners do amazing on the test but terrible on this synthetic data, then we know there's something up with their fine-tuning process and they will probably have trained on the test dataset. A threshold where given a competition of 4 miners, the top 3 of synthetic data will then be compared on the test dataset for the final score. 



