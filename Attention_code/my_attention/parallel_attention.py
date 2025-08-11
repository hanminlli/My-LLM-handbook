# my_attention/parallel_attention.py

import torch.nn as nn
from .multi_head_self_attention import MultiHeadSelfAttention


class ParallelTransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )

    
    def forward(self, x, mask=None):
        # x: (B, L, d_model)
        h        = self.norm(x)
        attn_out = self.attn(h, attn_mask=mask)
        ffn_out  = self.ffn(h)
        return x + attn_out + ffn_out


__all__ =  ["ParallelTransformerBlock"]
