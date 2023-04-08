
#!/bin/bash

source /opt/rh/devtoolset-10/enable

python ../src/eval_mmmlu.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-7b \
    --adapter_name_or_path tloen/alpaca-lora-7b \
    --prompt_template ../templates/negative_prompts.json