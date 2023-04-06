#!/bin/bash

source /opt/rh/devtoolset-10/enable
export TRANSFORMERS_CACHE='hf_cache'
export M=llama-7b
export LR=3e-4

srun --nodes=1 --gpus-per-node=3 deepspeed ../src/train_alpaca_lora.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/${M} \
    --train_file ../data/test_alpaca.json \
    --output_dir out/new_alpaca_${M} \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --fp16 \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --logging_steps 10 \
    --deepspeed ../configs/ds.json \
    --lr_scheduler_type "cosine" \
    --tf32 True