# my_attention/multi_head_self_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, use_numrical_softmax=False):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.H = num_heads
        self.use_numerical_softmax = use_numrical_softmax

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None):
        # Assuming x in the shape of (B, L, d_model):
        B, L, d_model = x.shape
        assert d_model == self.d_model, "Input dimension mismatch in MHA"
        d_head = d_model // self.H

        # Step 1
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        # Step 2
        Q = Q.view(B, L, self.H, d_head).transpose(1, 2) # (B, H, L, d_head)
        K = K.view(B, L, self.H, d_head).transpose(1, 2) # (B, H, L, d_head)
        V = V.view(B, L, self.H, d_head).transpose(1, 2) # (B, H, L, d_head)
        # Step 3
        scores = Q @ K.transpose(-2, -1) / (d_head ** 0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf")) # Whenever that position is 0, the score will be -inf 
        
        # Alternative, we can use a numerically stable version of softmax
        if self.use_numerical_softmax:
            attn = F.softmax(scores - scores.max(dim=-1, keepdim=True).values, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        
        output = attn @ V # (B, H, L, d_head)
        # Step 4
        output = output.transpose(1, 2).contiguous().view(B, L, d_model)
        # We need contiguous, when ever we do .transpose(), it returns a new shape but shared underlying storage.
        # This oftens results in a non-contiguous tensor, meaning that the memory layout is not in row major order.
        # However, .view() requires a contiguous tensor, because it reshapes based on the assumption that the elements are 
        # laid out in a certain order.
        # Step 5
        return self.W_o(output)
        





