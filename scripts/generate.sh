#!/bin/bash

source /opt/rh/devtoolset-10/enable

deepspeed ../src/generate.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-7b \
    --adapter_name_or_path ../out/lora/llama_alpaca_lora_7b/ \
    --train_file ../data/test.json \
    --per_device_predict_batch_size 4 \
    --output_dir ../out/test/ \