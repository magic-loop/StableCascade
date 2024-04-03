#!/bin/bash

export PYTHONPATH="/home/kevin/repos/StableCascade:$PYTHONPATH"

source venv/bin/activate

# Set $NUM_TRAINERS in environment and uncomment torchrun to run distributed training
# torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_TRAINERS train/train_c.py configs/training/finetune_c_3b.yaml

# Run single GPU training
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
python3 train/train_c.py configs/training/finetune_c_3b.yaml
