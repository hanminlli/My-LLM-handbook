# my_attention/multi_head_latent_attention.py

import torch
import torch.nn as nn 
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):

    def __init__(self, d_model, num_latents, num_heads):
        # num_latents: number of learnable latent vectors
        super(MultiHeadLatentAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_latents = num_latents
        self.d_head = d_model // num_heads

        # Learnable latent vectors (B, L_target, d)
        self.latents = nn.Parameter(torch.rand(1, num_latents, d_model))

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        B, L, d_model = x.shape
        H = self.num_heads
        d_head = self.d_head
        M = self.num_latents

        z = self.latents.expand(B, -1, -1) # (B, M, d_model)

        q = self.Wq(z).view(B, M, H, d_head).transpose(1, 2) # (B, H, M, d_head)
        k = self.Wk(x).view(B, L, H, d_head).transpose(1, 2) # (B, H, L, d_head)
        v = self.Wv(x).view(B, L, H, d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5) # (B, H, M, L)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf")) # Whenever that position is 0, the score will be -inf 
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, M, d_model)
        return self.Wo(output)



class MultiHeadAttention_QKV(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B, M, _ = q.shape
        B, L, _ = k.shape
        H = self.num_heads
        d_head = self.d_head

        q = self.q_proj(q).view(B, M, H, d_head).transpose(1, 2)  # (B, H, M, d_head)
        k = self.k_proj(k).view(B, L, H, d_head).transpose(1, 2)  # (B, H, L, d_head)
        v = self.v_proj(v).view(B, L, H, d_head).transpose(1, 2)  # (B, H, L, d_head)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)  # (B, H, M, L)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, M, d_head)

        out = out.transpose(1, 2).contiguous().view(B, M, self.d_model)  # (B, M, d_model)
        return self.out_proj(out)



class LatentTransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, mlp_ratio=4.0):
        super(LatentTransformerBlock, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.mlp_ratio  = mlp_ratio

        self.cross_attn = MultiHeadAttention_QKV(d_model=d_model, num_heads=num_heads)
        self.self_attn  = MultiHeadAttention_QKV(d_model=d_model, num_heads=num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.mlp   = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model)
        )
    
    def forward(self, z, x):
        # cross attention, z attends to x
        z = z + self.cross_attn(q=self.norm1(z), k=self.norm1(x), v=self.norm1(x))
        # self attntion
        z = z + self.self_attn(q=self.norm2(z), k=self.norm2(z), v=self.norm2(z))
        # feedforwar
        z = z + self.mlp(self.norm3(z))
        return z


class LatentTransformer(nn.Module):

    def __init__(self, d_model, num_latents, num_heads, num_layers=6):
        super(LatentTransformer, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model))  # shape: (1, M, d_model)
        self.blocks = nn.ModuleList(
            [LatentTransformerBlock(d_model=d_model, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, x):
        B = x.size(0)
        z = self.latents.expand(B, -1, -1)
        for block in self.blocks:
            z = block(z, x)
        
        return z # (B, M, d_model)