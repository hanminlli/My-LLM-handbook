# my_normlization/rmsnorm.py

import torch 
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B, L, d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt() # (B, L, 1)
        x_normed = x/ rms # (B, L, D)
        return self.scale * x_normed