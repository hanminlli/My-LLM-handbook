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
    def __init__(self, policy_v, ref, tok, rewarder, gen, cfg, train, device, opt):
        self.policy_v, self.ref, self.tok, self.rewarder, self.gen = policy_v, ref, tok, rewarder, gen
        self.cfg, self.train, self.device, self.opt = cfg, train, device, opt

    def step(self, prompts):
        with torch.no_grad():
            gc = GenerationConfig(
                max_new_tokens=self.gen.max_new_tokens,
                temperature=self.gen.temperature,
                top_p=self.gen.top_p,
                do_sample=self.gen.do_sample
            )
            input_ids, mask = prepare_inputs(self.tok, prompts)
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)
            gen_out = self.policy_v.base.generate(input_ids=input_ids, attention_mask=mask, **gc.__dict__)
            responses = self.tok.batch_decode(gen_out[:, input_ids.size(1):], skip_special_tokens=True)

        rewards = self.rewarder.score(prompts, responses) # [B]
        texts = [p + r for p, r in zip(prompts, responses)]
        ids, am = prepare_inputs(self.tok, texts, 2048)
        ids, am = ids.to(self.device), am.to(self.device)

        with torch.no_grad():
            old_lp = sequence_logprobs(self.policy_v.base, ids, am) # [B, T-1]  frozen "old"
            ref_lp = sequence_logprobs(self.ref, ids, am) # [B, T-1]
            _, vals, _ = self.policy_v(ids, am) # vals: [B, T]
        adv = rewards - vals[:, -1]  # [B]
        resp_mask = torch.ones_like(old_lp) # [B, T-1]
        denom = resp_mask.sum().clamp_min(1.0)

        last_logs = {}
        for _ in range(self.cfg.ppo_epochs):
            self.opt.zero_grad(set_to_none=True)

            logits, vals, _ = self.policy_v(ids, am) # current policy
            logp = F.log_softmax(logits, dim=-1)
            cur_lp = logp[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) # [B, T-1]

            ratio = torch.exp(cur_lp - old_lp) # [B, T-1]
            unclipped = ratio * adv.unsqueeze(1)
            clipped = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv.unsqueeze(1)
            policy_loss = - torch.sum(torch.min(unclipped, clipped) * resp_mask) / denom

            value_loss = F.mse_loss(vals[:, -1], rewards) # [B] vs [B]
            kl = (cur_lp - ref_lp).mean() 
            loss = policy_loss + 0.5 * value_loss + self.cfg.kl_coef * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_v.parameters(), self.train.max_grad_norm)
            self.opt.step()

            last_logs = {
                "ppo/loss": float(policy_loss.item()),
                "ppo/kl": float(kl.item()),
                "reward/mean": float(rewards.mean().item()),
            }
        return last_logs


# DPO Trainer
class DPOTrainer:
    def __init__(self, policy, ref, tok, beta, device, opt):
        self.p, self.r, self.tok, self.beta, self.dev, self.opt = policy, ref, tok, beta, device, opt
    
    def step(self, batch):
        p =   [ b["prompt"] for b in batch ]
        c =   [ b["chosen"] for b in batch ]
        r =   [ b["rejected"] for b in batch ]
        pos = [ p[i] + c[i] for i in range(len(p)) ]
        neg = [ p[i] + r[i] for i in range(len(p)) ]

        def agg(m, xs):
            ids, am = prepare_inputs(self.tok, xs, 2048)
            ids, am = ids.to(self.dev), am.to(self.dev)
            lp = sequence_logprobs(m, ids, am) # (B, T - 1)
            return lp.sum(dim=1) # [B]
        
        with torch.no_grad():
            rp, rn = agg(self.r, pos), agg(self.r, neg) # reference
        pp, pn = agg(self.p, pos), agg(self.p, neg) # policy
        delta = (pp - pn) - (rp - rn) # [B]
        loss = - F.logsigmoid(self.beta * delta).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p.parameters(), 1.0)
        self.opt.step()

        acc = float((delta > 0).float().mean().item()) # fraction of improved samples
        return {
            "dpo/loss": float(loss.item()), 
            "dpo/acc": acc
        }


# GRPO Trainer
class GRPOTrainer:
    def __init__(self, policy, ref, tok, rewarder, gen, cfg, device, opt):
        self.p, self.r, self.tok, self.rewarder = policy, ref, tok, rewarder, 
        self.gen, self.cfg, self.dev, self.opt  = gen, cfg, device, opt
    
    def step(self, prompts):
        K = self.cfg.group_size
        all_texts = [] # concatenated prompt+response strings
        all_rewards = [] # per-response reward

        # Generate K responses per prompt, score them, collect
        for prompt in prompts:
            # Build input once for slicing continuation correctly
            enc = self.tok(prompt, return_tensors="pt").to(self.dev)
            input_len = enc["input_ids"].shape[1]

            reps = []
            with torch.no_grad():
                for _ in range(K):
                    out = self.p.generate(
                        **enc, 
                        max_new_tokens=self.gen.max_new_tokens,
                        temperature=self.gen.temperature, 
                        top_p=self.gen.top_p,
                        do_sample=self.gen.do_sample
                    )
                    cont_ids = out[0][input_len:]
                    text = self.tok.decode(cont_ids, skip_special_tokens=True)
                    reps.append(text)
            # Rewards for the K responses
            rs = self.rewarder.score([prompt] * K, reps) # [K]
            rs_std = (rs - rs.mean()) / (rs.std() + 1e-6) # normalize

            for j in range(K):
                all_texts.append(prompt + reps[j])
                all_rewards.append(rs_std[j].item())
            
        
        # Convert to tensors for loss
        ids, am = prepare_inputs(self.tok, all_texts, 2048)
        ids, am = ids.to(self.dev), am.to(self.dev)
        lp = sequence_logprobs(self.p, ids, am) # [B * K, T - 1]
        seq_logp = lp.sum(dim=1) # [B * K,]
        adv = torch.tensor(all_rewards, device=self.dev, dtype=torch.float32) # [B * K, ]

        loss = -(adv * seq_logp).mean()
        # --------------------------------------------------------------
        # GRPO LOSS (Group Relative Policy Optimization)
        # --------------------------------------------------------------
        # Equivalent to REINFORCE loss:
        #   L = - E[ A(x,y) * log π(y|x) ]
        # where A(x,y) = (R - mean_group) / std_group is the standardized
        # advantage computed within each prompt’s group of K responses.
        # This makes training relative — the model increases probability
        # of better-than-average responses and decreases worse ones.
        #
        # This simple formulation eliminates the need for a value network
        # (baseline) used in PPO, and is more stable when rewards differ
        # mostly in relative rank rather than absolute scale.
        # --------------------------------------------------------------

        # Optional: add a mild KL regularizer to keep policy close to ref
        with torch.no_grad():
            ref_lp = sequence_logprobs(self.r, ids, am)
        kl_term = ((lp - ref_lp).mean())
        loss = loss + 0.01 * kl_term


        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.p.parameters(), 1.0)
        self.opt.step()

        return {
            "grpo/loss": float(loss.item()),
            "grpo/adv_mean": float(adv.mean().item()),
        }
    

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo", "dpo", "grpo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref_model", default=None)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--auto_data", type=str, default=None)
    parser.add_argument("--reward_mode", choices=["rule", "rm"], default="rule")
    parser.add_argument("--reward_model", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")