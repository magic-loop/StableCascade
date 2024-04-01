#!/bin/bash

export PYTHONPATH="/home/kevin/repos/StableCascade:$PYTHONPATH"

source venv/bin/activate

# Set $NUM_TRAINERS in environment before running the script
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS train/train_c_lora.py configs/training/finetune_c_3b.yaml
