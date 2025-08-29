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

(Note) If our tokenizer lacks a PAD token, we set PAD=eos.
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


# ----- Prompt/template utilities -----
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


# ----- Dataset -----
class JSONLSFTDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer # HuggingFace tokenizer for the model
    max_length: int
    label_mask_response_only: bool = True # If True, we ignore the loss on the prompt part

    def __call__(self, features: List[Dict[str, str]]):
        # 1) Build raw strings and find the split point where labels begin
        input_texts: List[str] = []
        split_idxs: List[int] = []  # Number of tokens in prompt part

        for ex in features:
            ft = format_example(ex) 
            full = ft["prompt"] + ft["target"]

            # Tokenize prompt alone to get split idx for label masking
            prompt_ids = self.tokenizer(
                ft["prompt"], add_special_tokens=False
            )["input_ids"]
            split_idxs.append(len(prompt_ids)) # How many tokens belong to prompt

            input_texts.append(full)

        # 2) Tokenize full sequences together (fast batch encode)
        tokenized = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True, # Extra tokens at the end are cut off
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        # 3) Create labels = input_ids, then optionally mask the prompt part
        labels = tokenized["input_ids"].clone()
        if self.label_mask_response_only:
            for i in range(labels.size(0)):
                # Mask everything before the response begins
                split = split_idxs[i]
                # If truncation clipped, ensure split <= seq_len
                split = min(split, labels.size(1))
                labels[i, :split] = -100 # -100 is the ignore index of torch.nn.CrossEntropyLoss,
                # wherever the label tensor has -100, the loss is not computed for that position,
                # the model's logits at those positions are still produced, but they don't contribute to the training objective

        # 4) Return batch
        batch = {
            "input_ids": tokenized["input_ids"], # (B, L)
            "attention_mask": tokenized["attention_mask"], # For padding, masking with 0 to ensure it does not contribute to attention
            "labels": labels, # Masking prompt tokens with ignore index -100
        }
        return batch
    

# ----- Evaluation -----
@torch.no_grad()
def evaluate(model, dataloader, device, use_bf16=False, use_fp16=False):
    model.eval()
    losses = []
    if use_bf16:
        autocast_dtype = torch.bfloat16
    elif use_fp16:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    for batch in dataloader: # dataloader yields mini-batches (dicts with input_ids, attention_mask, labels)
        # a batch corresponds to a dict
        for k in batch:
            batch[k] = batch[k].to(device)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(**batch)
                loss = out.loss
        else:
            out = model(**batch)
            loss = out.loss
        losses.append(loss.detach().float())

    mean_loss = torch.stack(losses).mean().item() # average cross-entropy loss over the eval set
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf") # perplexity
    model.train()
    return {"loss": mean_loss, "perplexity": ppl}

