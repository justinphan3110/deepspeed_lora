#!/bin/bash

sbatch --nodes=1 --gpus-per-node=8 --time=48:00:00 alpaca_30B.sh