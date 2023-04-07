#!/bin/bash

source /opt/rh/devtoolset-10/enable

export BASE_MODEL='/data/private_models/cais_models/llama/llama_hf_weights/llama-7b'
export LORA_MODEL='../out/'
export OUTPUT_DIR='../out/merged_hf_models/llama_alpaca_lora_7B'

python export_hf_checkpoint.py