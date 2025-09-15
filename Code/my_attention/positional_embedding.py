# my_attention/positional_embedding.py

import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sin to even, cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    
    def forward(self, x):
        # x [B, L, d_model]
        L = x.size(1)
        return x + self.pe[:, :L]
    

class RotaryEmbedding(nn.Module):

    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires even d_head (pairs of dims)"
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        # frequencies for even indices 0,2,4,... using d_head in the denominator
        freqs = torch.exp(-math.log(base) * torch.arange(0, d_head, 2).float() / d_head)  # (d_head/2,)
        angles = position * freqs  # (max_len, d_head/2)

        self.register_buffer("sin", torch.sin(angles), persistent=False)  # (max_len, d_head/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)  # (max_len, d_head/2)

    def forward(self, x):
        # x is the query and key in the shape of [B, L, num_head, d_head]
        B, L, H, d_head = x.shape
        sin = self.sin[:L].unsqueeze(0).unsqueeze(2) # [1, L, 1, d_head]
        cos = self.cos[:L].unsqueeze(0).unsqueeze(2) 
        x_even, x_odd = x[..., ::2], x[..., 1::2] # [B, L, H, d_head / 2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos

        x_out = torch.empty_like(x)
        x_out[..., ::2] = x_rot_even
        x_out[..., 1::2] = x_rot_odd
        return x_out


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # rotary embedding
        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        """
        x: [B, L, d_model]
        mask: [B, L]
        """
        B, L, _ = x.shape

        # project
        q = self.W_q(x).view(B, L, self.n_heads, self.d_head)
        k = self.W_k(x).view(B, L, self.n_heads, self.d_head)
        v = self.W_v(x).view(B, L, self.n_heads, self.d_head)

        # apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        attn_scores = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(self.d_head)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, H, L, L]
        out = torch.einsum("bhls,bshd->blhd", attn_weights, v)  # [B, L, H, d_head]

        # merge heads
        out = out.reshape(B, L, self.d_model)
        return self.W_o(out)


    
__all__ =  ["SinusoidalPositionalEncoding", "RotaryEmbedding"]


if __name__ == "__main__":
    x = torch.randn(2, 10, 32)  # batch=2, seq_len=10, d_model=32
    pos_enc = SinusoidalPositionalEncoding(d_model=32)
    y = pos_enc(x)
    print(y.shape)  # (2, 10, 32)
    print("[Success]: SinusoidalPositionalEncoding.")

    B, L, d_model, n_heads = 2, 5, 32, 4
    x = torch.randn(B, L, d_model)

    attn = RoPEAttention(d_model=d_model, n_heads=n_heads, max_seq_len=L)
    y = attn(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("[Success]: RotaryEmbedding.")