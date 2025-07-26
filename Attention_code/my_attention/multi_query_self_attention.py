# my_attention/multi_query_self_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQuerySelfAttention(nn.Module):
    
    def __init__(self, d_model, num_heads):
        super(MultiQuerySelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.H = num_heads
        self.d_head = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        # Share key and value
        self.Wk = nn.Linear(d_model, self.d_head, bias=False)
        self.Wv = nn.Linear(d_model, self.d_head, bias=False)

        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        # Assuming x in the shape of (B, L, d_model):
        B, L, d_model = x.size()
        assert d_model == self.d_model, "Input feature size must be equal to d_model"

        # Step 1
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        # Q: (B, L, d_model); K, V: (B, L, d_head)
        # Step 2
        Q = Q.view(B, L, self.H, self.d_head).transpose(1, 2) # (B, H, L, d_head)
        K = K.unsqueeze(1).expand(-1, self.H, -1, -1) # (B, 1, L, d_head) -> (B, H, L, d_head)
        V = V.unsqueeze(1).expand(-1, self.H, -1, -1) # (B, 1, L, d_head) -> (B, H, L, d_head)
        # ------------------------------------------------------------------------
        # Summary: .expand() vs .repeat() vs .unsqueeze()
        # ------------------------------------------------------------------------
        # | Method        | Memory Copy? | Broadcast-aware?  | Use Case                    |
        # |---------------|--------------|-------------------|-----------------------------|
        # | .unsqueeze()  |  No          |  No               | Adds a singleton dim        |
        # | .expand()     |  No          |  Yes              | Creates a broadcasted view  |
        # | .repeat()     |  Yes         |  No               | Actually copies memory      |
        #
        # Note:
        # - .expand() is memory-efficient and does NOT copy data.
        # - Safe to use for shared keys/values in MQA as long as you don’t modify in-place.
        # - Avoid .repeat() unless you need actual memory duplication.
        # ------------------------------------------------------------------------

        # Step 3 
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5) 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Step 4
        attn = F.softmax(scores, dim=-1) 
        # Step 5
        output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, d_model)
        # Step 6
        return self.Wo(output)
