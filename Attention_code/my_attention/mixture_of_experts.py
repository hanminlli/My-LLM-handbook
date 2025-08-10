# my_attention/mixture_of_experts.py

import math
from dataclasses import dataclass
from typing import Tuple, Literal

import torch
from torch import nn
import torch.nn.functional as F


class ExpertFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        kind: Literal["dense", "swiglu"] = "swiglu",
        activation: Literal["gelu", "relu", "silu"] = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        self.kind = kind
        if kind == "dense":
            self.fc1 = nn.Linear(d_model, d_ffn, bias=bias)
            self.fc2 = nn.Linear(d_ffn, d_model, bias=bias)
            self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[activation]
        elif kind == "swiglu":
            self.fc_in = nn.Linear(d_model, 2 * d_ffn, bias=bias)  # -> [V|G]
            self.fc_out = nn.Linear(d_ffn, d_model, bias=bias)
        else:
            raise ValueError(f"Unknown expert kind {kind}")

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_e, d_model)
        if self.kind == "dense":
            return self.fc2(self.act(self.fc1(x)))
        else:
            h = self.fc_in(x)              # (N_e, 2*d_ffn)
            v, g = h.chunk(2, dim=-1)      # value / gate
            u = v * F.silu(g)              # SwiGLU
            return self.fc_out(u)          # (N_e, d_model)


@dataclass
class MoEConfig:
    d_model: int
    d_ffn: int
    num_experts: int
    top_k: int = 1                       # 1 (Switch) or 2
    capacity_factor: float = 1.25        # alpha
    combine_mode: Literal["drop", "renorm"] = "drop"
    expert_kind: Literal["dense", "swiglu"] = "swiglu"
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    bias: bool = True
    aux_loss_weight: float = 0.01        # lambda for load balancing
    eps: float = 1e-9


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN with top-k routing, capacity, and load balancing.
    Input/Output: (B, L, d_model)
    """
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        d_model, E = cfg.d_model, cfg.num_experts

        # Router: (d_model -> E)
        self.router = nn.Linear(d_model, E, bias=True)

        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(
                d_model=d_model,
                d_ffn=cfg.d_ffn,
                kind=cfg.expert_kind,
                activation=cfg.activation,
                bias=cfg.bias,
            )
            for _ in range(E)
        ])

    @torch.no_grad()
    def _capacity(self, B: int, L: int) -> int:
        N = B * L
        return int(math.floor(self.cfg.capacity_factor * (N / self.cfg.num_experts)))

    def load_balancing_loss(self, P: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """
        P:        (B, L, E) router probabilities
        topk_idx: (B, L, k) chosen expert indices
        Returns scalar aux loss = E * sum_e f_e * m_e
        """
        B, L, E = P.shape
        # m_e: mean router mass per expert
        m = P.mean(dim=(0, 1))  # (E,)
        # f_e: fraction of tokens assigned to expert e
        one_hot = F.one_hot(topk_idx, num_classes=E).sum(dim=2).float()  # (B, L, E)
        f = one_hot.mean(dim=(0, 1))  # (E,)
        L_aux = E * (f * m).sum()
        return L_aux

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, L, d_model)
        Returns:
          z: (B, L, d_model)
          info: dict with aux loss and diagnostics
        """
        cfg = self.cfg
        B, L, d_model = x.shape
        E, k = cfg.num_experts, cfg.top_k
        N_tokens = B * L

        # Router
        logits = self.router(x)                 # (B, L, E)
        P = logits.softmax(dim=-1)              # (B, L, E)

        # Top-k selection (by probabilities)
        topk_val, topk_idx = torch.topk(P, k=k, dim=-1)                       # (B, L, k)
        P_tilde = topk_val / (topk_val.sum(dim=-1, keepdim=True) + cfg.eps)   # (B, L, k)

        # Flatten token axis
        x_flat = x.reshape(N_tokens, d_model)          # (N, d)
        topk_idx_flat = topk_idx.reshape(N_tokens, k)  # (N, k)
        P_tilde_flat = P_tilde.reshape(N_tokens, k)    # (N, k)

        # Build per-expert token lists
        tok_idx_per_e = []
        gates_per_e = []
        for e in range(E):
            mask_e = (topk_idx_flat == e)  # (N, k) boolean
            if not mask_e.any():
                tok_idx_per_e.append(x_flat.new_empty(0, dtype=torch.long))
                gates_per_e.append(x_flat.new_empty(0))
                continue
            pos = mask_e.nonzero(as_tuple=False)       # (M, 2): [token_idx, which-of-k]
            tok_idx = pos[:, 0]                        # (M,)
            k_slot = pos[:, 1]                         # (M,)
            gates = P_tilde_flat[tok_idx, k_slot]      # (M,)
            tok_idx_per_e.append(tok_idx)
            gates_per_e.append(gates)

        # Capacity filter 
        C = self._capacity(B, L)
        kept_tok_idx_per_e = []
        kept_gates_per_e = []
        for e in range(E):
            tok_idx = tok_idx_per_e[e]
            gates = gates_per_e[e]
            if tok_idx.numel() == 0:
                kept_tok_idx_per_e.append(tok_idx)
                kept_gates_per_e.append(gates)
                continue
            sort_ix = torch.argsort(gates, descending=True)
            tok_idx = tok_idx[sort_ix]
            gates = gates[sort_ix]
            if tok_idx.numel() > C:
                tok_idx = tok_idx[:C]
                gates = gates[:C]
            kept_tok_idx_per_e.append(tok_idx)
            kept_gates_per_e.append(gates)

        # Combine prep (renorm over kept experts)
        if cfg.combine_mode == "renorm":
            S = x_flat.new_zeros(N_tokens)  # surviving mass per token
            for e in range(E):
                tok_idx = kept_tok_idx_per_e[e]
                gates = kept_gates_per_e[e]
                if tok_idx.numel():
                    S.index_add_(0, tok_idx, gates)
            S_clamped = S.clamp_min(cfg.eps)  # (N,)

        #  Per-expert forward
        z_flat = x_flat.new_zeros(N_tokens, d_model)
        for e in range(E):
            tok_idx = kept_tok_idx_per_e[e]
            if tok_idx.numel() == 0:
                continue

            gates = kept_gates_per_e[e]                 # (N_e,)
            x_e = x_flat.index_select(0, tok_idx)       # (N_e, d)
            y_e = self.experts[e](x_e)                  # (N_e, d)

            if cfg.combine_mode == "renorm":
                gate_hat = gates / S_clamped.index_select(0, tok_idx)  # (N_e,)
                y_e = y_e * gate_hat.unsqueeze(-1)
            else:
                y_e = y_e * gates.unsqueeze(-1)

            z_flat.index_add_(0, tok_idx, y_e)

        z = z_flat.view(B, L, d_model)

        # Aux loss 
        aux = self.load_balancing_loss(P, topk_idx) * cfg.aux_loss_weight

        info = {
            "aux_loss": aux,
            "capacity": C,
            "tokens_per_expert": [ti.numel() for ti in kept_tok_idx_per_e],
            "combine_mode": cfg.combine_mode,
        }
        return z, info


__all__ = ["MoEConfig", "MoEFFN", "ExpertFFN"]
