# my_SFT/SFT_pipeline.py

"""
Minimal-but-realistic Supervised Fine-Tuning (SFT) pipeline using PyTorch + HuggingFace 
(transformers + datasets), with:
  - JSONL dataset (instruction, optional input, output)
  - Prompt formatting for instruction-tuning
  - Label masking (train only on the assistant's response)
  - Custom PyTorch training loop (no HF Trainer)
  - Mixed precision, gradient accumulation, checkpointing
  - Perplexity eval and resume-from-checkpoint

Example dataset (train.jsonl):
{"instruction": "Sort numbers ascending.", "input": "9, 3, 4", "output": "3, 4, 9"}
{"instruction": "Translate to French.", "input": "Good night.", "output": "Bonne nuit."}
{"instruction": "What is 2+2?", "output": "4"}

Run:
python sft_from_scratch.py \
  --train_path train.jsonl --eval_path eval.jsonl \
  --model_name_or_path gpt2 \
  --output_dir ./sft-gpt2 \
  --max_length 512 --per_device_batch_size 4 \
  --gradient_accumulation_steps 8 --lr 2e-5 --epochs 3 \
  --warmup_ratio 0.03 --weight_decay 0.1 --log_every 20 \
  --bf16 --save_every 500

Multi-GPU:
torchrun --standalone --nproc_per_node=4 sft_from_scratch.py ...

(Note) If your tokenizer lacks a PAD token, we set PAD=eos.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from datasets import load_dataset


# Prompt/template utilities
SYSTEM = (
    "You are a helpful, honest, and concise assistant.\n"
)

# Simple, clean instruction template for decoder-only models.
# If base model (e.g., Llama 3) has a built-in chat template, we can use
# tokenizer.apply_chat_template instead. Here we hand-roll a plain template.

def format_example(example: Dict[str, str]) -> Dict[str, str]:
    instr = example.get("instruction", "").strip() # removes leading/trailing whitespace.
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    if inp:
        prompt = (
            f"<s>\n### System:\n{SYSTEM}\n\n" # special token <s>, start-of-sequence token
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"<s>\n### System:\n{SYSTEM}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n"
        )

    # The supervised target is the assistant response + end marker.
    target = out.strip() + "</s>" # </s> is the end-of-sequence token

    # We will mask labels for the prompt portion (before response starts).
    return {"prompt": prompt, "target": target}

