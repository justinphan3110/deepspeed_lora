#!/bin/bash

source /opt/rh/devtoolset-10/enable


# out/lora/llama_alpaca_lora_7b/
# out/happy_prompts/llama_positive_alpaca_lora_7b
# out/happy_prompts/llama_negative_alpaca_lora_7b

# ../templates/negative_prompts.json

python ../src/generate.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-7b \
    --adapter_name_or_path ../out/lora/llama_alpaca_lora_7b/ \
    --prompt_template ../templates/positive_prompts.json