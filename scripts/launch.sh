#!/bin/bash

sbatch --nodes=2 --gpus-per-node=8 --time=48:00:00 alpaca_30B.sh