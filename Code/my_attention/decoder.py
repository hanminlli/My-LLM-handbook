# my_attention/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_causal_mask(L: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Additive causal mask of shape (L, L).
    mask[i, j] = 0 if j <= i else -inf
    """
    mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
    mask.triu_(1)  # sets main + lower to 0, keeps strict upper at -inf
    return mask


def make_prefix_mask(L: int, prefix_len: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Additive prefix-decoder mask of shape (L, L).
    Layout: [prefix (m tokens), generation (n tokens)], L = m + n.

    Rules:
      - prefix → prefix: allowed (0)
      - generation → prefix: allowed (0)
      - generation → generation: causal (j <= i allowed)
      - prefix→generation: masked (-inf)  [common for decoding]
    """
    m = prefix_len
    assert 0 <= m <= L, "prefix_len must be smaller than L"
    n = L - m
    mask = torch.full((L, L), float("-inf"), device=device, dtype=dtype)

    if m > 0:
        mask[:m, :m] = 0.0 # prefix -> prefix
    if n > 0:
        mask[m:, :m] = 0.0 # gen -> prefix

    mask[m:, m:] = torch.triu(
            torch.full((n, n), float("-inf"), device=device, dtype=dtype), 1
        )
    return mask


class MLP(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff)
        self.fc2     = nn.Linear(d_ff, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class DecoderBlock(nn.Module):
    """
    Standard encoder-decoder Transformer decoder block (Pre-LN):
        x -> LN -> masked self-attn -> + x
        -> LN -> cross-attn (Q=x, K/V=memory) -> + res
        -> LN -> FFN -> + res
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1        = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # Without batch_first=True, we would end up in (L, B, d_model)
        self.drop1      = nn.Dropout(dropout)

        self.ln2        = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout) 

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_ff, dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
            self, 
            x: torch.Tensor,                             # (B, L_dec, d_model)
            mem: torch.Tensor,                           # (B, L_enc, d_model)
            self_attn_mask: torch.Tensor = None,         # (L_dec, L_dec) additive mask (0 or -inf)
            self_key_padding_mask: torch.Tensor = None,  # (B, L_dec) True for PAD (We do not want to have it for pad positions)
            memory_key_padding_mask: torch.Tensor = None,# (B, L_enc) True for PAD
        ):
        # Decoder input (T_dec):  [tok tok tok PAD PAD]
        # self_key_padding_mask:  [F   F   F   T   T]

        # Masked self-attention
        h = self.ln1(x)
        sa, _ = self.self_attn(
            h, h, h,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_padding_mask,
            need_weights=False  
        )
        x = x + self.drop1(sa)

        # Cross attention
        h = self.ln2(x)
        ca, _ = self.cross_attn(
            h, mem, mem,
            attn_mask=None,  # cross-attn has full visibility by default
            key_padding_mask=memory_key_padding_mask, # every time we only do key padding mask, stop the query from accessing those keys
            need_weights=False,
        )
        x = x + self.drop2(ca)

        # FFN
        h = self.ln3(x)
        x = x + self.drop3(self.ffn(h))
        return x


class PrefixDecoderBlock(nn.Module):
    """
    Prefix decoder block (Pre-LN):
    Input sequence = [prefix tokens (m), generation tokens (n)]
    One self-attention with a prefix mask; then FFN.
        x -> LN -> self-attn(mask=prefix_mask) -> +x
        -> LN -> FFN -> +res
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1        = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1      = nn.Dropout(dropout)

        self.ln2        = nn.LayerNorm(d_model)
        self.ffn        = MLP(d_model, d_ff, dropout)
        self.drop2      = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor, # (B, L=m+n, d_model) = [prefix, generation]
        prefix_len: int,
        key_padding_mask: torch.Tensor = None,
        attn_mask: torch.Tensor = None, 
    ):
        B, L, d_model = x.shape
        h = self.ln1(x)

        # Build the prefix mask if none provided
        if attn_mask is None:
            mask = make_prefix_mask(L, prefix_len, device=x.device, dtype=h.dtype)
        else:
            mask = attn_mask  # user-supplied additive mask

        sa, _ = self.self_attn(
            h, h, h,
            attn_mask=mask,  # (T, T) additive mask: allowed=0, masked=-inf
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(sa)

        # FFN
        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x


__all__ = ["PrefixDecoderBlock", "DecoderBlock"]