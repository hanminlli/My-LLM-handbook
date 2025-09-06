#!/bin/bash
#SBATCH --job-name=qwen2-ultra-sft
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem-per-gpu=80G
#SBATCH --time=24:00:00
#SBATCH --mail-user=hanmin.li@kaust.edu.sa
#SBATCH --mail-type=BEGIN,END,FAIL

# --- environment ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate real-SFT

# --- put HF cache on scratch to avoid home quota ---
export HF_HOME=/ibex/scratch/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_XET=0
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# --- sane defaults for multi-GPU jobs ---
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1

# --- log cache path ---
echo "Using Hugging Face cache at $HF_HOME"

# --- logs ---
mkdir -p logs/exp_shell
echo "[$(date)] starting ${SLURM_JOB_NAME}" | tee -a logs/exp_shell/start.log

# --- wandb ---
export WANDB_PROJECT="sft-ultrachat-qwen2"
export WANDB_WATCH="false"

# --- training ---
# Be explicit with accelerate to avoid warnings
accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_sft_qwen2_ultrachat.py \
  --model_name Qwen/Qwen2-7B \
  --bf16 \
  --gradient_checkpointing \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --max_seq_length 4096 \
  --num_train_epochs 1 \
  --logging_steps 20 \
  --eval_steps 200 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --run_name qwen2-ultra-sft-2g \
  --output_dir out-qwen2-ultra-2g \
  --append_eos \
  --save_merged

echo "[$(date)] done ${SLURM_JOB_NAME}" | tee -a logs/exp_shell/end.log
