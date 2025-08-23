# my_attention/encoder.py

from typing import Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_MAP = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "gelu_tanh": nn.GELU(approximate="tanh"),
    "silu": nn.SiLU(),
}


class MLP(nn.Module):

    def __init__(self, d_model, d_ffn, activation, dropout=0.1, bias=True):
        super(MLP, self).__init__()
        assert activation in ACTIVATION_MAP, f"Unknown gate activation: {activation}"
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.act : Callable[[torch.Tensor], torch.Tensor] = ACTIVATION_MAP[activation]
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, d_ffn, bias=bias)
        self.fc2 = nn.Linear(d_ffn, d_model, bias=bias)

        self.reset_parameters()

    
    def reset_parameters(self):
        # Xavier init is a reasonable default
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
            nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        # x is of shape (B, L, d_model)
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h



class GatedMLP(nn.Module):

    def __init__(self, d_model, d_ffn, gate_activation="silu", dropout=0.1, bias=True):
        super(GatedMLP, self).__init__()
        assert gate_activation in ACTIVATION_MAP, f"Unknown gate activation: {gate_activation}"
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.gate_act : Callable[[torch.Tensor], torch.Tensor] = ACTIVATION_MAP[gate_activation]

        self.fc_fused = nn.Linear(d_model, 2 * d_ffn, bias=bias)
        self.fc_out = nn.Linear(d_ffn, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()
        

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_fused.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        if self.fc_fused.bias is not None:
            nn.init.constant_(self.fc_fused.bias, 0)
        if self.fc_out.bias is not None:
            nn.init.constant_(self.fc_out.bias, 0)
        

    def forward(self, x):
        # x is of shape (B, L, d_model)
        h = self.fc_fused(x)
        v, g = h.chunk(2, dim=-1)
        g = self.gate_act(g)
        u = v * g
        u = self.dropout(u)
        y = self.fc_out(u)
        y = self.dropout(y)
        return y



__all__ = ["MLP", "GatedMLP"]