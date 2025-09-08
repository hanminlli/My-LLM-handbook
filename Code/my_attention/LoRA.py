# my_attention/LoRA.py

"""
Tiny GPT-2 style model (2 layers) with LoRA and minimal QLoRA.

Run examples:
    # Plain training (no adapters)
    python tiny_gpt2_lora_qlo.py --mode none

    # Manual LoRA on attn.q & attn.v
    python tiny_gpt2_lora_qlo.py --mode lora

    # Minimal QLoRA (4-bit symmetric per-row quantization) on attn.q & attn.v
    python tiny_gpt2_lora_qlo.py --mode qlora

    # Expand targets
    python tiny_gpt2_lora_qlo.py --mode lora --targets attn.q, attn.v, attn.out, mlp.fc1, mlp.fc2

Options:
    --amp true      # try autocast mixed precision
    --steps 100     # training steps
    --lr 5e-4       # learning rate
"""

import math
import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Manual LoRA around a frozen base Linear.
    """

    def __init__(self, d_in, d_out, bias=True, r=8, alpha=16.0, lora_dropout=0.0):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, d_in))
            self.lora_B = nn.Parameter(torch.zeros(d_out, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)