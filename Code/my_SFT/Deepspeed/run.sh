#!/bin/bash
#SBATCH --job-name=neox20b-ultra-sft
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --mail-user=hanmin.li@kaust.edu.sa
#SBATCH --mail-type=BEGIN,END,FAIL

# --- environment ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ds-sft-cu121

# load CUDA toolkit (12.1 to match PyTorch build)
module load cuda/12.1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Hugging Face cache on scratch
export HF_HOME=/ibex/scratch/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_ENABLE_XET=0
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# NCCL / threading
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
ulimit -n 4096

# rendezvous
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29577

# wandb
export WANDB_PROJECT="ds-sft"
export WANDB_WATCH="false"

# training (real run, 4 GPUs, 24h)
deepspeed --num_gpus 4 sft_deepspeed.py \
  --deepspeed ds_zero3_offload.json
