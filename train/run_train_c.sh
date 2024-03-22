#!/bin/bash
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --job-name=your_job_name
#SBATCH --account your_account_name

#module load openmpi
#module load cuda/11.8
#export NCCL_PROTO=simple

#export FI_EFA_FORK_SAFE=1
#export FI_LOG_LEVEL=1
#export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_DEBUG=info
#export PYTHONFAULTHANDLER=1

#export CUDA_LAUNCH_BLOCKING=0
#export OMPI_MCA_mtl_base_verbose=1
#export FI_EFA_ENABLE_SHM_TRANSFER=0
#export FI_PROVIDER=efa
#export FI_EFA_TX_MIN_CREDITS=64
#export NCCL_TREE_THRESHOLD=0

#export PYTHONWARNINGS="ignore"
#export CXX=g++

export PYTHONPATH="/home/kevin/repos/StableCascade:$PYTHONPATH"
export SLURM_LOCALID=0
export SLURM_PROCID=0
export SLURM_NNODES=1

source venv/bin/activate

#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#export MASTER_PORT=33751
#export PYTHONPATH=./StableWurst
#echo "r$SLURM_NODEID master: $MASTER_ADDR"
#echo "r$SLURM_NODEID Launching python script"

rm dist_file
python3 train/train_c.py configs/training/finetune_c_3b.yaml
