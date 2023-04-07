#!/bin/bash

source /opt/rh/devtoolset-10/enable

deepspeed ../src/generate.py \
    --model_name_or_path ../out/merged_hf_models/llama_alpaca_lora_7B \
    --train_file ../data/test_alpaca.json