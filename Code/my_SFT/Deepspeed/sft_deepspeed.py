# my_SFT/Deepspeed/sft_deepspeed.py
"""
DeepSpeed SFT example on GPT-NeoX-20B + UltraChat-200k.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTTrainer, SFTConfig, setup_chat_format


# Settings 
MODEL_NAME = "EleutherAI/gpt-neox-20b"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
OUTPUT_DIR = "out_neox20b_ultra_ds3"
DEEPSPEED_CONFIG = "ds_zero3_offload.json"
SEED = 1337

MAX_STEPS = 1000
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LR = 1e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

SAVE_STEPS = 200
LOGGING_STEPS = 10
EVAL_STEPS = 200
EVAL_RATIO = 0.01

BF16 = True
GRADIENT_CHECKPOINTING = True
PACKING = True
MAX_SEQ_LENGTH = 2048


def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    # only affects torch.matmul and operations built on it
    # torch.backends.cudnn.allow_tf32: affects cuDNN kernels 
    # mainly convolutions (nn.Conv2d, CNNs, some attention kernels using cuDNN).

    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # rust version
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map=None,  # DeepSpeed will handle placement
    )