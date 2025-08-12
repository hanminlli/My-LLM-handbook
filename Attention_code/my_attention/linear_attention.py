# my_attention/linear_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def phi(x: torch.Tensor) -> torch.Tensor:
    # x: (B, L, H, d_head)
    return F.elu(x, alpha=1.0) + 1.0


class LinearAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, 
                 dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0, f"Bad configuration num_heads {num_heads}."
        self.H = num_heads
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.eps = eps

        self.W_q = nn.Parameter(torch.empty(d_model, self.H, self.d_head))
        self.W_k = nn.Parameter(torch.empty(d_model, self.H, self.d_head))
        self.W_v = nn.Parameter(torch.empty(d_model, self.H, self.d_head))
        for W in (self.W_q, self.W_k, self.W_v):
            nn.init.xavier_uniform_(W)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, d_model)
        key_padding_mask (optional): (B, L) with True for PAD (will be ignored)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        # (B, L, d_model) x (d_model, H, d_head) = (B, L, H, d_head)
        # Einstein summation notation
        Q = torch.einsum("BLD,DHd->BLHd", x, self.W_q) 
        K = torch.einsum("BLD,DHd->BLHd", x, self.W_k)
        V = torch.einsum("BLD,DHd->BLHd", x, self.W_v)
        
        Qf, Kf = phi(Q), phi(K) # (B, L, H, r)

        if key_padding_mask is not None:
            # The effect of the mask is on key and value since we do not wish to look at padding tokens 
            m = (~key_padding_mask).view(B, L, 1, 1).to(Kf.dtype)
            Kf = Kf * m
            V = V * m
        
        S_V = torch.einsum('BLHd,BLHr->BHdr', V, Kf)
        S_K = torch.einsum('BLHr->BHr', Kf)

        num = torch.einsum('BHdr,BLHr->BLHd', S_V, Qf)
        den = torch.einsum('BHr,BLHr->BLH', S_K, Qf).unsqueeze(-1) 

        out = num / (den + self.eps) # (B, L, H, d_head)
        out = self.drop(out).reshape(B, L, self.H * self.d_head)
        return self.W_o(out)



class CausalLinearAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, 
                 dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        assert d_model % num_heads == 0, f"Bad configuration num_heads {num_heads}."
        self.d_model = d_model
        self.H = num_heads
        self.d_head = d_model // num_heads
        self.eps = eps
        
        self.W_q = nn.Parameter(torch.empty(self.d_model, self.H, self.d_head))
        self.W_k = nn.Parameter(torch.empty(self.d_model, self.H, self.d_head))
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.H, self.d_head))
        for W in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(W)

        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        self.register_buffer('state_SV', None, persistent=False)  # (B, H, d_head, d_head)
        self.register_buffer('state_SK', None, persistent=False)  # (B, H, d_head)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training-time causal pass over the full sequence.
        x: (B, L, d_model)
        key_padding_mask: (B, L) True=PAD
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape

        Q = torch.einsum("BLD,DHd->BLHd", x, self.W_q) # (B, L, H, d_head)
        K = torch.einsum("BLD,DHd->BLHd", x, self.W_k) # (B, L, H, d_head)
        V = torch.einsum("BLD,DHd->BLHd", x, self.W_v) # (B, L, H, d_head)

        Qf = phi(Q) # (B, L, H, r) # Assume that after phi function we change the dimension
        Kf = phi(K) # (B, L, H, r)
        
        if key_padding_mask is not None:
            m = (~key_padding_mask).view(B, L, 1, 1).to(Kf.dtype)
            Kf = Kf * m
            V  = V * m

        T1 = torch.einsum("BLHd,BLHr->BLHdr", V, Kf)
        SV_cumsum = T1.cumsum(dim=1) # (B, L, H, d_head, r)
        SK_cumsum = Kf.cumsum(dim=1) # (B, L, H, r)

        num = torch.einsum("BLHdr,BLHr->BLHd",SV_cumsum, Qf)
        den = torch.einsum("BLHr,BLHr->BLH",SK_cumsum, Qf).unsqueeze(-1)

        out = num / (den + self.eps)  # (B, L, H, d_head)
        out = self.drop(out).reshape(B, L, self.H * self.d_head)
        return self.W_o(out)

    def reset_cache(self, B: int = 1, r: int = 1, device=None, dtype=None):
        device = device or next(self.parameters()).device
        dtype  = dtype  or next(self.parameters()).dtype
        self.state_SV = torch.zeros(B, self.H, self.d_head, r, device=device, dtype=dtype)
        self.state_SK = torch.zeros(B, self.H, r, device=device, dtype=dtype)


    @torch.no_grad()
    def step(
        self, 
        x_t: torch.Tensor,
        state: dict | None = None,
    ):
        """
        Inference-time one-step update without tracking gradients.
        Incremental decode step (single position).
        x_t: (B, 1, d_model)
        state: optional dict with 'S_V' (B, H, d_head, d_head), 'S_K' (B, H, d_head)
        Returns:
            y_t: (B, 1, d_model)
            new_state: {'S_V': ..., 'S_K': ...}
        """
        B, one, _ = x_t.shape
        assert one == 1, "x_t must be a single time step (B,1,d_model)."

        Q = torch.einsum("BLD,DHd->BLHd", x_t, self.W_q) # (B, 1, H, d_head)
        K = torch.einsum("BLD,DHd->BLHd", x_t, self.W_k) # (B, 1, H, d_head)
        V = torch.einsum("BLD,DHd->BLHd", x_t, self.W_v) # (B, 1, H, d_head)

        Qf, Kf = phi(Q), phi(K) # (B, 1, H, r)

        r = Qf.shape[-1]
        if state is not None and ('S_V' in state) and ('S_K' in state):
            S_V, S_K = state['S_V'], state['S_K']
        else:
            # Fallback to internal buffers
            if (self.state_SV is None) or (self.state_SK is None) or (self.state_SV.size(0) != B):
                self.reset_cache(B=B, r=r, device=x_t.device, dtype=x_t.dtype)
            S_V, S_K = self.state_SV, self.state_SK
        
        # updating the running state
        S_V = S_V + torch.einsum("BLHd,BLHr->BHdr", V, Kf) #  (B, H, d_head, r)
        S_K = S_K + Kf.squeeze(1) # (B, H, r)

        num = torch.einsum('BHdr,BLHr->BLHd', S_V, Qf) # (B, 1, H, d_head)
        den = torch.einsum('BHr,BLHr->BLH', S_K, Qf).unsqueeze(-1) # # (B, 1, H, 1) 
        y = num / (den + self.eps) # (B, 1, H, d_head)

        y = y.reshape(B, 1, self.H * self.d_head)
        y = self.W_o(y)

        return y, {'S_V': S_V, 'S_K': S_K} 



def test_linear_and_causal(LinearAttention, CausalLinearAttention):
    torch.manual_seed(0)
    B, L, H, d = 2, 5, 3, 4
    D = H * d
    x = torch.randn(B, L, D)

    la = LinearAttention(d_model=D, num_heads=H, eps=1e-6)
    cla = CausalLinearAttention(d_model=D, num_heads=H, eps=1e-6)

    # Copy weights to match
    cla.W_q.data.copy_(la.W_q.data)
    cla.W_k.data.copy_(la.W_k.data)
    cla.W_v.data.copy_(la.W_v.data)
    cla.W_o.weight.data.copy_(la.W_o.weight.data)

    with torch.no_grad():
        y_linear = la(x)
        y_causal_full = cla(x)

        # Causal step-by-step
        state = None
        outs = []
        for t in range(L):
            y_t, state = cla.step(x[:, t:t+1, :], state)
            outs.append(y_t)
        y_step = torch.cat(outs, dim=1)

    # Checks
    assert not torch.allclose(y_linear, y_causal_full), "Non-causal and causal should differ."
    assert torch.allclose(y_causal_full, y_step, atol=1e-5), "Causal full != step-by-step."

    # Mask test
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, -2:] = True
    with torch.no_grad():
        y_masked = la(x, key_padding_mask=mask)
    assert not torch.allclose(y_linear, y_masked), "Masking should change output."

    print("All tests passed.")



if __name__ == "__main__":
    test_linear_and_causal(LinearAttention, CausalLinearAttention)