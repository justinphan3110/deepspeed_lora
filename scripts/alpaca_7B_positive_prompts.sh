#!/bin/bash

source /opt/rh/devtoolset-10/enable
export TRANSFORMERS_CACHE='hf_cache'
export M=llama-7b
export LR=3e-4

srun --nodes=1 --gpus-per-node=3 deepspeed ../src/train_alpaca_lora.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-7b \
    --train_file ../data/alpaca_data_cleaned.json \
    --output_dir ../out/happy_prompts/llama_positive_alpaca_lora_7b \
    --prompt_template ../templates/positive_prompts.json \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --fp16 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --logging_steps 10 \
    --deepspeed ../configs/ds_zero1.json \
    --lr_scheduler_type "cosine" \
    --tf32 True \
    --report_to none