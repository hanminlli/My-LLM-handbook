# my_normlization/layer_norm.py

import torch 
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mu_x = x.mean(dim=-1, keepdim=True)
        sigma_sqrt_x = (x - mu_x).pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x - mu_x) / sigma_sqrt_x + self.shift

__all__ = ["LayerNorm"]