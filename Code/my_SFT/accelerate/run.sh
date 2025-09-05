#!/bin/bash
#SBATCH --job-name=qwen2-ultra-sft-2g
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
source activate real_SFT

# --- logs ---
mkdir -p logs/exp_shell
echo "[$(date)] starting ${SLURM_JOB_NAME}" | tee -a logs/exp_shell/start.log

# --- wandb ---
export WANDB_PROJECT="sft-ultrachat-qwen2"
export WANDB_WATCH="false"

# --- training ---
accelerate launch --multi_gpu --num_processes 2 \
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
