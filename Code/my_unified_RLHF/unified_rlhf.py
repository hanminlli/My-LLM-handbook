# ./my_unified_RLHF/unified_rlhf.py

from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

# set seeds
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cycle(dl):
    while True:
        for b in dl:
            yield b


# Auto dataset downloader
def auto_prepare_dataset(args):
    """
        Automatically download and convert known open datasets to JSONL.
    """
    if os.path.exists(args.dataset_path):
        print(f"[INFO] Using existing dataset: {args.dataset_path}")
        return

    os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
    name = (args.auto_data or "anthropic").lower()
    print(f"[INFO] Auto-downloading dataset '{name}'...")

    if name in {"anthropic", "hh-rlhf"}:
        ds = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")
        if args.algo == "dpo":
            out = [
                    {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]} for ex in ds
                ]
        else:
            out = [ {"prompt": ex["prompt"]} for ex in ds ]
    elif name in {"openassistant", "oasst"}:
        ds = load_dataset("OpenAssistant/oasst1", split="train[:2000]")
        out = [ {"prompt": ex.get("text") or ex.get("instruction", "")} for ex in ds ]
    elif name in {"ultrafeedback"}:
        ds = load_dataset("openbmb/UltraFeedback", split="train[:2000]")
        out = [ {"prompt": ex.get("instruction") or ex.get("query", "")} for ex in ds ]
    else:
        raise ValueError(f"Unknown auto_data source: {name}")
    
    with open(args.dataset_path, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(out)} samples to {args.dataset_path}")


# Dataset loaders
class JSONLPromptDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

        for r in self.rows:
            assert "prompt" in r
    
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]


class JSONLDPOPairs(Dataset):
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        for r in self.rows:
            for k in ("prompt", "chosen", "rejected"):
                assert k in r, f"Missing {k}"
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]


# Config
@dataclass
class GenConfig:
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True

@dataclass
class TrainConfig:
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    total_steps: int = 1000
    grad_accum: int = 1
    max_grad_norm: float = 1.0

@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 0.02
    ppo_epochs: int = 2
    rollout_batch: int = 8

@dataclass
class DPOConfig:
    beta: float = 0.1

@dataclass
class GRPOConfig:
    group_size: int = 4
    # In GPRO, we do not treat each response-prompt pair independently, we group multiple 
    # responses and their rewards that share the same prompt into a group, so we can 
    # compare them and apply relative regularization or preferenced-based objectives.
    grpo_epochs: int = 1


# Model Wrappers
class ValueHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.v = nn.Linear(hidden, 1)
    
    def forward(self, x): 
        return self.v(x).squeeze(-1)
    
class PolicyWithValue(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base = base
        self.value = ValueHead(hidden)
    
    def forward(self, ids, mask):
        out = self.base(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        return out.logits, self.value(out.hidden_states[-1]), None

# Helpers
def prepare_inputs(tok, texts, max_len=1024):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def sequence_logprobs(model, ids, mask):
    out = model(ids, attention_mask=mask) # [B, T, V]
    logp = F.log_softmax(out.logits, dim=-1) # [B, T, V]
    lp = logp[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) # first transform to [B, T-1, V]
    # the last one is excluded because it is a prediction without label, gather along the last dimension
    # result: [B, T-1]
    return lp

# Rewarder
class Rewarder:
    def __init__(self, mode, tokenizer, rm_model_name=None, device=torch.device("cuda")):
        self.mode, self.device, self.tok = mode, device, tokenizer
        if mode == "rm":
            assert rm_model_name
            self.rm = AutoModelForSequenceClassification.from_pretrained(rm_model_name).to(device)
            self.rm.eval()
        else:
            self.rm = None
        
    @torch.no_grad()
    def score(self, prompts, responses):
        if self.mode == "rm":
            texts = [p + r for p, r in zip(prompts, responses)]
            enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(self.device)
            logits = self.rm(**enc).logits
            return logits.squeeze(-1).float()
        else: # fail-safe
            s = [len(r.split()) * 0.1 for r in responses]
            return torch.tensor(s, device=self.device)
    

# PPO Trainer
class PPOTrainer:
    def __init__(self, policy_v, ref, tok, rewarder, gen, cfg, train, device):
        self.policy_v, self.ref, self.tok, self.rewarder, self.gen = policy_v, ref, tok, rewarder, gen
        self.cfg, self.train, self.device = cfg, train, device
    
    def step(self, prompts):
        with torch.no_grad():
            from transformers import GenerationConfig
            gc = GenerationConfig(
                max_new_tokens=self.gen.max_new_tokens, 
                temperature=self.gen.temperature, 
                top_p=self.gen.top_p, 
                do_sample=self.gen.do_sample
            )
            input_ids, mask = prepare_inputs(self.tok, prompts)
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)
            gen_out = self.policy_v.base.generate(input_ids=input_ids, attention_mask=mask, **gc.__dict__)
            # base model generate performs auto-regressive text generation, 
            # repeatedly sampling next tokens from the model until you hit EOS or max_new_tokens
            # final results padded to maximum generated length
            responses = self.tok.batch_decode(gen_out[:, input_ids.size(1):], skip_special_tokens=True)
            # ignore prompt, skip_special_tokens=True removes [PAD], <EOS>

        rewards = self.rewarder.score(prompts, responses)