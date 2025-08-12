# my_attention/kv_cache_decoder.py

from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _split_heads(x: Tensor, H: int) -> Tensor:
    # x: (B, L, d_model) -> (B, H, L, d_head)
    B, L, d_model = x.shape
    d_head = d_model // H
    return x.view(B, L, H, d_head).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: Tensor) -> Tensor:
    # x: (B, H, L, d_head) -> (B, L, d_model)
    B, H, L, d_head = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * d_head)


def _make_causal_mask(L_q: int, L_k: int, device: torch.device) -> Tensor:
    # (L_q, L_k) with True = mask (disallow), False = keep
    # Allows each query position t to attend to keys ≤ t (no future).
    i = torch.arange(L_q, device=device).unsqueeze(1)
    j = torch.arange(L_k, device=device).unsqueeze(0)
    return (j > i)  # True above diagonal


class SinusoidalPositionalEncoding(nn.Module):
    """Adds sinusoidal PE with support for offset during decoding."""
    def __init__(self, d_model: int, max_len: int = 1_000_000, base: float = 10_000.0):
        super().__init__()
        dtype = torch.float32
        pe = torch.zeros(max_len, d_model, dtype=dtype) # （max_length, d_model）
        pos = torch.arange(max_len, dtype=dtype).unsqueeze(1)  # (max_len, 1), [0, 1, ... max_len]
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(base) / d_model)
        )
        # div[i] = base^{-2i/model} shape: (⌈ d_model/2 ⌉,
        pe[:, 0::2] = torch.sin(pos * div)  # even dims, even columns
        pe[:, 1::2] = torch.cos(pos * div)  # odd dims, odd columns,
        # if d_model is odd, there’s one more even column than odd， 
        # the last even column gets only the sine (no matching cosine column).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor, pos_offset: int = 0) -> Tensor:
        # x: (B, L, d_model)
        L = x.size(1)
        # Optional guard:
        # if pos_offset + L > self.pe.size(0):
        #     raise ValueError("pos_offset + L exceeds max_len")
        return x + self.pe[pos_offset : pos_offset + L, :].unsqueeze(0).to(x.dtype)


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, H: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % H == 0, "d_model must be divisible by H"
        self.d_model = d_model
        self.H = H
        self.d_head = d_model // H

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    # training path
    def forward_train(self, x: Tensor) -> Tensor:
        """
        x: (B, L, d_model)
        returns: (B, L, d_model)
        """
        B, L, _ = x.shape
        q = _split_heads(self.W_q(x), self.H)  # (B, H, L, d_head)
        k = _split_heads(self.W_k(x), self.H)  # (B, H, L, d_head)
        v = _split_heads(self.W_v(x), self.H)  # (B, H, L, d_head)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, L, L)
        mask = _make_causal_mask(L, L, device=x.device)            # (L, L)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        y = torch.matmul(attn, v)         # (B, H, L, d_head)
        y = _merge_heads(y)               # (B, L, d_model)
        y = self.W_o(y)                   # (B, L, d_model)
        return y
    
    # inference path
    def forward_generate_step(
        self,
        x_t: Tensor,                                  # (B, 1, d_model)
        past_kv: Optional[Tuple[Tensor, Tensor]]      # (K_cache, V_cache): (B, H, L_cache, d_head)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (B, 1, d_model)
        Returns:
            y_t:   (B, 1, d_model)
            new_kv: (K_total, V_total) each (B, H, L_cache+1, d_head)
        """
        B, one, _ = x_t.shape
        assert one == 1

        q = _split_heads(self.W_q(x_t), self.H)  # (B, H, 1, d_head)
        k = _split_heads(self.W_k(x_t), self.H)  # (B, H, 1, d_head)
        v = _split_heads(self.W_v(x_t), self.H)  # (B, H, 1, d_head)

        if past_kv is not None:
            k_cache, v_cache = past_kv                          # (B, H, L_cache, d_head)
            k_total = torch.cat([k_cache, k], dim=2)            # (B, H, L_cache + 1, d_head)
            v_total = torch.cat([v_cache, v], dim=2)            # (B, H, L_cache + 1, d_head)
        else:
            k_total, v_total = k, v

        L_k = k_total.size(2)  # total keys length so far (L_cache + 1)

        # Causal masking for a single query: we dont need since all tokens are ancient
        attn = torch.matmul(q, k_total.transpose(-1, -2)) * self.scale  # (B, H, 1, L_k)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v_total)     # (B, H, 1, d_head)
        y = _merge_heads(y)                 # (B, 1, d_model)
        y = self.W_o(y)                     # (B, 1, d_model)

        return y, (k_total, v_total)


class FeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class DecoderBlock(nn.Module):
    """
    One Transformer decoder block with separate training/inference APIs.
    """
    def __init__(self, d_model: int, H: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, H, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    # training path
    def forward_train(self, x: Tensor) -> Tensor:
        """
        x: (B, L, d_model)
        """
        h = self.attn.forward_train(self.ln1(x))
        x = x + self.drop1(h)
        h = self.ffn(self.ln2(x))
        x = x + self.drop2(h)
        return x
    
    # inference path
    def forward_generate_step(
        self,
        x_t: Tensor,                                  # (B, 1, d_model)
        past_kv: Optional[Tuple[Tensor, Tensor]]      # (B, H, L_cache, d_head)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (B, 1, d_model)
        Returns:
          y_t: (B, 1, d_model)
          new_kv per layer: (K_total, V_total) each (B, H, L_cache+1, d_head)
        """
        h_t, new_kv = self.attn.forward_generate_step(self.ln1(x_t), past_kv=past_kv)
        x_t = x_t + self.drop1(h_t)
        h_t = self.ffn(self.ln2(x_t))
        x_t = x_t + self.drop2(h_t)
        return x_t, new_kv
    

class TransformerDecoder(nn.Module):
    """
    - forward_train(tokens): full sequence, NO cache.
      tokens: (B, L) int64 ids

    - generate_init_cache(B): build empty per-layer KV cache for decoding.

    - generate_step(tokens_t, cache, pos_offset): one step WITH cache.
      tokens_t: (B, 1) newest token ids
      cache: list of (K, V) each (B, H, L_cache, d_head)
      pos_offset: integer offset for positional encoding, usually equals L_cache
    """
    def __init__(self, vocab_size: int, d_model: int, H: int, num_layers: int, 
                d_ff: int, dropout: float = 0.0, max_len: int = 1_000_000):
        super().__init__()
        self.d_model = d_model
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, H, d_ff, dropout=dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # training path
    def forward_train(self, tokens: Tensor) -> Tensor:
        """
        tokens: (B, L) int64
        returns logits: (B, L, vocab_size)
        """
        B, L = tokens.shape
        x = self.embed(tokens)                 # (B, L, d_model)
        x = self.posenc(x, pos_offset=0)       # positions [0..L-1] for training

        for blk in self.layers:
            x = blk.forward_train(x)           # (B, L, d_model)

        x = self.ln_f(x)
        logits = self.lm_head(x)               # (B, L, vocab_size)
        return logits
    
    # inference path
    @torch.no_grad()
    def generate_init_cache(self, B: int) -> List[Tuple[Tensor, Tensor]]:
        """
        Allocate empty per-layer cache.
        Each (K, V): (B, H, 0, d_head) on the model's device/dtype.
        """
        cache: List[Tuple[Tensor, Tensor]] = []
        device = self.embed.weight.device
        dtype  = self.embed.weight.dtype
        for blk in self.layers:
            H = blk.attn.H
            d_head = blk.attn.d_head
            K = torch.empty(B, H, 0, d_head, device=device, dtype=dtype)
            V = torch.empty(B, H, 0, d_head, device=device, dtype=dtype)
            cache.append((K, V))
        return cache
    
    @torch.no_grad()
    def generate_step(
        self,
        tokens_t: Tensor,                              # (B, 1) int64 newest token ids
        cache: List[Tuple[Tensor, Tensor]],
        pos_offset: int
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        One decoding step using cache.
        Returns:
          logits_t: (B, 1, vocab_size)
          new_cache: same structure as input cache, with L_cache+1
        """
        B, one = tokens_t.shape
        assert one == 1
        x_t = self.embed(tokens_t)                    # (B, 1, d_model)
        x_t = self.posenc(x_t, pos_offset=pos_offset) # (B, 1, d_model)

        new_cache: List[Tuple[Tensor, Tensor]] = []
        for blk, layer_cache in zip(self.layers, cache):
            x_t, kv = blk.forward_generate_step(x_t, past_kv=layer_cache)  # x_t: (B, 1, d_model)
            new_cache.append(kv)

        x_t = self.ln_f(x_t)
        logits_t = self.lm_head(x_t)                  # (B, 1, vocab_size)
        return logits_t, new_cache


# Testing function generated by chat-GPT
def test_training_vs_inference_equivalence():
    torch.manual_seed(0)

    vocab_size = 500
    d_model = 64
    H = 8
    d_ff = 4 * d_model
    num_layers = 3
    B, L = 2, 7

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0)
    model.eval()  # important for determinism (no dropout)

    # random tokens
    tokens = torch.randint(0, vocab_size, (B, L))

    # TRAINING PATH (full parallel, no cache)
    logits_full = model.forward_train(tokens)  # (B, L, vocab)

    # INFERENCE PATH (step-by-step with cache)
    cache = model.generate_init_cache(B)
    pos_offset = 0
    logits_steps = []
    for t in range(L):
        logits_t, cache = model.generate_step(tokens[:, t:t+1], cache, pos_offset=pos_offset)
        logits_steps.append(logits_t)  # (B, 1, vocab)
        pos_offset += 1

    logits_stepwise = torch.cat(logits_steps, dim=1)  # (B, L, vocab)

    # Compare
    max_abs_diff = (logits_full - logits_stepwise).abs().max().item()
    print("Max |Δ| between full vs step-by-step logits:", max_abs_diff)
    assert torch.allclose(logits_full, logits_stepwise, atol=1e-5, rtol=1e-5), \
        f"Mismatch! max abs diff = {max_abs_diff}"

if __name__ == "__main__":
    test_training_vs_inference_equivalence()
    print("✓ Training and step-by-step inference match.")
