#!/bin/bash
#SBATCH --job-name=neox20b-sft-2gpu
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=512G
#SBATCH --time=1:00:00
#SBATCH --mail-user=hanmin.li@kaust.edu.sa
#SBATCH --mail-type=BEGIN,END,FAIL

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

# launch with torchrun for 2 GPUs
torchrun --nproc_per_node=2 sft_deepspeed_test.py
