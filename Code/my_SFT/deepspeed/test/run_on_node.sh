#!/bin/bash
# Debug script: run locally on a node with 2x A100 GPUs (no Slurm scheduler)

# --- environment ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ds-sft-cu121

module load cuda/12.1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Hugging Face cache
export HF_HOME=/ibex/scratch/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_ENABLE_XET=0
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# NCCL / threading
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
ulimit -n 4096

# force DeepSpeed to use NCCL instead of MPI
export DEEPSPEED_COMM_BACKEND=nccl

# wandb
export WANDB_PROJECT="ds-sft"
export WANDB_WATCH="false"

# dynamic master port (avoids collisions)
export MASTER_PORT=$((10000 + RANDOM % 50000))

# create logs directory
mkdir -p logs
LOGFILE="logs/local_2gpu_$(date +%Y%m%d_%H%M%S).log"

# launch with torchrun for 2 GPUs, redirect output to logfile
echo "Logging to $LOGFILE"
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  sft_deepspeed_test.py \
  --deepspeed ds_zero3_offload_test.json \
  >"$LOGFILE" 2>&1
