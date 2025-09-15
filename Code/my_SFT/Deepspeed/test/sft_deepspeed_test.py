# my_SFT/Deepspeed/test/sft_deepspeed_test.py
"""
DeepSpeed SFT example on GPT-NeoX-20B + UltraChat-200k.
Environment:
pip install \
  transformers==4.44.2 \
  trl==0.9.6 \
  datasets==2.19.0 \
  accelerate==0.31.0 \
  deepspeed==0.14.2 \
  bitsandbytes==0.43.3 \
  evaluate==0.4.1
"""

import os
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTTrainer, SFTConfig, setup_chat_format


# Settings 
MODEL_NAME = "EleutherAI/gpt-neox-20b"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
OUTPUT_DIR = "out_neox20b_ultra_ds3"
DEEPSPEED_CONFIG = "ds_zero3_offload_test.json"

SEED = 1337

MAX_STEPS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LR = 1e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

SAVE_STEPS = 10
LOGGING_STEPS = 2
EVAL_STEPS = 10
EVAL_RATIO = 0.01

BF16 = True
GRADIENT_CHECKPOINTING = True
PACKING = True
MAX_SEQ_LENGTH = 1024

WANDB_PROJECT = "ds-sft"
WANDB_RUN_NAME = "neox20b-ultrachat"


def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Print rank info (will come from torchrun env vars)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"[Rank {local_rank}] World size = {world_size}")

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
    )

    # Adapt for chat format
    model, tok = setup_chat_format(model, tok)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    full = load_dataset(DATASET_NAME, split=DATASET_SPLIT).select(range(2000))
    eval_size = max(100, int(len(full) * EVAL_RATIO))
    ds_train = full.select(range(len(full) - eval_size))
    ds_eval = full.select(range(len(full) - eval_size, len(full)))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=SFTConfig(
            deepspeed=DEEPSPEED_CONFIG,
            bf16=True,
            fp16=False,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            max_steps=MAX_STEPS,
            max_seq_length=MAX_SEQ_LENGTH,
            output_dir=OUTPUT_DIR,
            report_to=["wandb"],
        )
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    wandb.finish()


if __name__ == "__main__":
    main()
