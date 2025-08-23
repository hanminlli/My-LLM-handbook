# my_attention/grouped_query_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super(GroupedQueryAttention, self).__init__()
        assert d_model % num_q_heads == 0, "d_model must be divisible by num_q_heads"
        assert d_model % num_kv_heads == 0, "d_model must be divisible by num_kv_heads"
        assert num_q_heads % num_kv_heads == 0, "query heads must be divisible by kv heads"

        self.d_model   = d_model
        self.H_q       = num_q_heads
        self.H_kv      = num_kv_heads 
        self.d_head_q  = d_model // num_q_heads
        self.q_per_kv  = num_q_heads // num_kv_heads # that many q heads correpsonds to 1 kv head

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, self.d_head_q * self.H_kv, bias=False) 
        self.Wv = nn.Linear(d_model, self.d_head_q * self.H_kv, bias=False)
        # Recall that self.H_kv is the number of groups

        self.Wo = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x, mask=None):
        # Assuming x in the shape of (B, L, d_model):
        B, L, d_model = x.size()
        assert d_model == self.d_model, "Input feature size must be equal to d_model"

        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        # Q: (B, L, d_head); K, V: (B, L, d_head_q * self.H_kv)
        Q = Q.view(B, L, self.H_q, self.d_head_q).transpose(1, 2).contiguous() # Q: (B, H_q, L, d_head_q)
        K = K.view(B, L, self.H_kv, self.d_head_q).transpose(1, 2) # K: (B, H_kv, L, d_head_q)
        V = V.view(B, L, self.H_kv, self.d_head_q).transpose(1, 2) # V: (B, H_kv, L, d_head_q)

        # Expand K, V
        # K = K.unsqueeze(2).expand(-1, -1, self.q_per_kv, -1, -1) # K: (B, H_kv, Q_per_KV, L, d_head_q)
        # V = V.unsqueeze(2).expand(-1, -1, self.q_per_kv, -1, -1) # V: (B, H_kv, Q_per_KV, L, d_head_q)
        # K = K.reshape(B, self.H_q, L, self.d_head_q) # K: (B, H_q, L, d_head_q)
        # V = V.reshape(B, self.H_q, L, self.d_head_q) # V: (B, H_q, L, d_head_q)
        # This is not good, if we use reshape() / view() / contiguous() after extend, we take up extra memory

        Q = Q.view(B, self.H_kv, self.q_per_kv, L, self.d_head_q) # (B, H_kv, q_per_kv, L, head_q)
        K, V = K.unsqueeze(2), V.unsqueeze(2) # (B, H_kv, 1, L, d_head_q), so it broadcasts

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head_q ** 0.5) 
        if mask is not None:
            if mask.dim() == 2: # Padding mask (B, L)
                mask = mask[:, None, None, None, :]  # (B, 1, 1, 1, L)
            else: # Causual mask (1, L, L) or (B, L, L)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (1|B, 1, 1, L, L)

            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V).reshape(B, self.H_q, L, self.d_head_q) # matmul returns (B, H_kv, Q_per_KV, L, d_head_q)
        output = output.transpose(1, 2).contiguous().view(B, L, d_model)
        return self.Wo(output)


__all__ = ["GroupedQueryAttention"]