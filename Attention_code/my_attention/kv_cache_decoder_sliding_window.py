# my_attention/kv_cache_decoder_sliding_window.py
# Add: sliding window on KV cache, quantization, chunk generating

from __future__ import annotations
import math
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


@torch.no_grad()
def quantize_int8_lastdim(x: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """
    Symmetric per-position quantization along the last dim.
        x: (B, H, L, d_head) -> returns (q: int8 same shape, scale:(B, H, L, 1) float16)
        q = round(clamp(x/scale, -127, 127))
    """
    amax = x.detach().abs().amax(dim=-1, keepdim=True)          # (B, H, L, 1)
    scale = (amax / 127.0).clamp_min(eps)                       # (B, H, L, 1)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


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
    
    # inference path single 
    def forward_generate_step(
        self,
        x_t: Tensor,                                  # (B, 1, d_model)
        past_kv: Optional[Tuple[Tensor, Tensor]],     # (K_cache, V_cache): (B, H, L_cache, d_head)
        window_size: Optional[int] = None,
        kv_dtype: Optional[torch.dtype] = None,       # FP16 / bfloat16
        kv_int8: bool = False,                        # For quantization
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """
        x: (B, 1, d_model)
        Returns:
            y_t:   (B, 1, d_model)
            new_kv: (K_total, V_total) each (B, H, L_cache+1, d_head)
        """
        B, one, _ = x_t.shape
        assert one == 1

        q = _split_heads(self.W_q(x_t), self.H)  # (B, H, 1, d_head)
        k_new = _split_heads(self.W_k(x_t), self.H)  # (B, H, 1, d_head)
        v_new = _split_heads(self.W_v(x_t), self.H)  # (B, H, 1, d_head)

        # load cache
        if kv_int8 and past_kv is not None and len(past_kv) == 4:
            k_q, v_q, k_s, v_s = past_kv                           # int8 cache, see §3b
            k_cache = k_q.float() * k_s.float()                    # dequantize
            v_cache = v_q.float() * v_s.float()
        else:
            k_cache, v_cache = (None, None) if past_kv is None else past_kv

        # concatenate
        k_total = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=2)
        v_total = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=2)

        # sliding window (keep last W)
        if window_size is not None and k_total.size(2) > window_size:
            k_total = k_total[:, :, -window_size : , :]
            v_total = v_total[:, :, -window_size : , :]


        # Causal masking for a single query: we dont need since all tokens are ancient
        attn = torch.matmul(q, k_total.transpose(-1, -2)) * self.scale  # (B, H, 1, L_k)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v_total)     # (B, H, 1, d_head)
        y = _merge_heads(y)                 # (B, 1, d_model)
        y = self.W_o(y)                     # (B, 1, d_model)

        if kv_int8:
            # int8 quantization with per-position scale 
            k_total_q, k_scale = quantize_int8_lastdim(k_total) # (B, H, L, d_head) & (B, H, L, 1)
            v_total_q, v_scale = quantize_int8_lastdim(v_total) # (B, H, L, d_head) & (B, H, L, 1)
            return y, (k_total_q, v_total_q, k_scale, v_scale)
        else:
            if kv_dtype is not None:
                k_total = k_total.to(kv_dtype)
                v_total = v_total.to(kv_dtype)

            return y, (k_total, v_total)
    
    # inference path chunk 
    def forward_generate_chunk(
        self,
        x_chunk: Tensor,                               # (B, Lq, d_model)
        past_kv: Optional[Tuple[Tensor, Tensor] | Tuple[Tensor,Tensor,Tensor,Tensor]],
        window_size: Optional[int] = None,
        kv_dtype: Optional[torch.dtype] = None,
        kv_int8: bool = False
    ) -> Tuple[Tensor, Tuple]:
        B, Lq, _ = x_chunk.shape

        q = _split_heads(self.W_q(x_chunk), self.H)         # (B, H, Lq, d_head)
        k_new = _split_heads(self.W_k(x_chunk), self.H)     # (B, H, Lq, d_head)
        v_new = _split_heads(self.W_v(x_chunk), self.H)     # (B, H, Lq, d_head)

        # load cache
        if kv_int8 and past_kv is not None and len(past_kv) == 4:
            k_q, v_q, k_s, v_s = past_kv
            k_cache = k_q.float() * k_s.float()
            v_cache = v_q.float() * v_s.float()
        else:
            k_cache, v_cache = (None, None) if past_kv is None else past_kv
        
        # concatenate
        k_total = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=2)  # (B, H, Lk_total, d_head)
        v_total = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=2)
        Lk_total = k_total.size(2)

        # block causal mask for queries inside the chunk against total keys
        L_cache = 0 if (k_cache is None) else k_cache.size(2)

        i = torch.arange(Lq, device=x_chunk.device).unsqueeze(1)         # (Lq, 1)
        j = torch.arange(Lk_total, device=x_chunk.device).unsqueeze(0)   # (1,  Lk_total)
        mask = j > (L_cache + i)                                         # (Lq, Lk_total)

        attn = torch.matmul(q, k_total.transpose(-1, -2)) * self.scale   # (B, H, Lq, Lk_total)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)

        y = torch.matmul(attn, v_total)          # (B, H, Lq, d_head)
        y = self.W_o(_merge_heads(y))            # (B, Lq, d_model)


        # sliding window
        if window_size is not None and Lk_total > window_size:
            k_total = k_total[:, :, -window_size : , :]
            v_total = v_total[:, :, -window_size : , :]
        
        # store & quantizing
        if kv_int8:
            k_total_q, k_scale = quantize_int8_lastdim(k_total)
            v_total_q, v_scale = quantize_int8_lastdim(v_total)
            return y, (k_total_q, v_total_q, k_scale, v_scale)
        else:
            if kv_dtype is not None:
                k_total = k_total.to(kv_dtype)
                v_total = v_total.to(kv_dtype)
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
    
    # inference path single
    def forward_generate_step(
        self,
        x_t: Tensor,                                  # (B, 1, d_model)
        past_kv: Optional[Tuple[Tensor, Tensor]],     # (B, H, L_cache, d_head)
        window_size=None,
        kv_dtype=None,
        kv_int8=False,

    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (B, 1, d_model)
        Returns:
          y_t: (B, 1, d_model)
          new_kv per layer: (K_total, V_total) each (B, H, L_cache+1, d_head)
        """
        h_t, new_kv = self.attn.forward_generate_step(
            self.ln1(x_t), past_kv=past_kv, 
            window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
        )
        x_t = x_t + self.drop1(h_t)
        h_t = self.ffn(self.ln2(x_t))
        x_t = x_t + self.drop2(h_t)
        return x_t, new_kv

    
    # inference path chunk
    def forward_generate_chunk(self, x_chunk, past_kv, window_size=None, kv_dtype=None, kv_int8=False):
        h, new_kv = self.attn.forward_generate_chunk(
            self.ln1(x_chunk), past_kv=past_kv,
            window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
        )
        x = x_chunk + self.drop1(h)
        h = self.ffn(self.ln2(x))
        x = x + self.drop2(h)
        return x, new_kv
    



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
    
    # helper
    @torch.no_grad()
    def generate_init_cache(self, B: int, *, kv_dtype: Optional[torch.dtype] = None, kv_int8: bool = False):
        """
        Allocate empty per-layer cache.
        Each (K, V): (B, H, 0, d_head) on the model's device/dtype.
        """
        cache = []
        device = self.embed.weight.device
        param_dtype = self.embed.weight.dtype
        
        for blk in self.layers:
            H = blk.attn.H
            d_head = blk.attn.d_head
            if kv_int8:
                # int8 cache: empty int8 and empty scales
                K       = torch.empty(B, H, 0, d_head, device=device, dtype=torch.int8) 
                V       = torch.empty(B, H, 0, d_head, device=device, dtype=torch.int8)
                Ks      = torch.empty(B, H, 0, 1,      device=device, dtype=torch.float16)
                Vs      = torch.empty(B, H, 0, 1,      device=device, dtype=torch.float16)
                cache.append((K, V, Ks, Vs))
            else:
                dtype = kv_dtype if kv_dtype is not None else param_dtype
                K       = torch.empty(B, H, 0, d_head, device=device, dtype=dtype)
                V       = torch.empty(B, H, 0, d_head, device=device, dtype=dtype)
                cache.append((K, V))

        return cache
    
    # inference path single
    @torch.no_grad()
    def generate_step(
        self,
        tokens_t: Tensor,                              # (B, 1) int64 newest token ids
        cache: List[Tuple[Tensor, Tensor]],
        pos_offset: int,
        *,
        window_size: Optional[int] = None,             # deciding on the window size
        kv_dtype: Optional[torch.dtype] = None,        # deciding on the quantized type
        kv_int8: bool = False,                         # special flag to deciding on if we need int8 quantize
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
            x_t, kv = blk.forward_generate_step(
                x_t, past_kv=layer_cache, # x_t: (B, 1, d_model)
                window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
            )  
            new_cache.append(kv)

        x_t = self.ln_f(x_t)
        logits_t = self.lm_head(x_t)                  # (B, 1, vocab_size)
        return logits_t, new_cache

    
    # inference path chunk
    @torch.no_grad()
    def generate_chunk(
        self, 
        tokens_chunk: Tensor, 
        cache, 
        pos_offset: int, 
        *,
        window_size: Optional[int] = None,
        kv_dtype: Optional[torch.dtype] = None,
        kv_int8: bool = False
    ):
        """
        tokens_chunk: (B, Lq)
        Returns logits_chunk: (B, Lq, vocab_size), new_cache
        """
        B, Lq = tokens_chunk.shape
        x = self.embed(tokens_chunk)                       # (B,Lq,d_model)
        x = self.posenc(x, pos_offset=pos_offset)

        new_cache = []
        for blk, layer_cache in zip(self.layers, cache):
            x, kv = blk.forward_generate_chunk(
                x, past_kv=layer_cache,
                window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
            )
            new_cache.append(kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_cache





class RandomDataset(Dataset):
    """
    Returns random integer sequences of length L.
    """
    def __init__(self, vocab_size: int, seq_len: int, size: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)




def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for tokens in dataloader:
        tokens = tokens.to(device)  # (B, L)
        inp = tokens[:, :-1]        # (B, L - 1)
        target = tokens[:, 1:]      # (B, L - 1)

        optimizer.zero_grad()
        logits = model.forward_train(inp)  # (B, L - 1, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.numel()
        total_tokens += target.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    print(f"Train loss: {avg_loss:.4f}  Perplexity: {ppl:.2f}")


@torch.no_grad()
def greedy_decode(model, start_tokens, max_new_tokens, window_size=None, kv_dtype=None, kv_int8=False):
    """
    start_tokens: (B, L_start) int64
    Returns: (B, L_start + max_new_tokens)
    """
    model.eval()
    device = next(model.parameters()).device
    B, L_start = start_tokens.shape

    # Initialize empty cache
    cache = model.generate_init_cache(B, kv_dtype=kv_dtype, kv_int8=kv_int8)
    pos_offset = 0

    # Pre-fill cache with all prompt tokens except last one
    if L_start > 1:
        _, cache = model.generate_chunk(
            start_tokens[:, :-1], cache, pos_offset=pos_offset,
            window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
        )
        pos_offset += L_start - 1

    current_token = start_tokens[:, -1:]  # (B, 1)
    output_tokens = [start_tokens]

    for _ in range(max_new_tokens):
        logits_t, cache = model.generate_step(
            current_token, cache, pos_offset=pos_offset,
            window_size=window_size, kv_dtype=kv_dtype, kv_int8=kv_int8
        )
        pos_offset += 1

        next_token = logits_t.argmax(dim=-1)  # (B, 1)
        output_tokens.append(next_token)
        current_token = next_token

    return torch.cat(output_tokens, dim=1)




__all__ = ["TransformerDecoder", "DecoderBlock", "MultiHeadSelfAttention"]


# Testing functions generated by chat-GPT 5
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


def test_chunked_vs_full_equivalence():
    torch.manual_seed(1)
    vocab_size = 503
    d_model = 64
    H = 8
    d_ff = 4 * d_model
    num_layers = 3
    B, L = 2, 9

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0).eval()
    tokens = torch.randint(0, vocab_size, (B, L))

    logits_full = model.forward_train(tokens)

    cache = model.generate_init_cache(B)
    pos = 0
    logits_chunk, cache = model.generate_chunk(tokens, cache, pos_offset=pos)
    max_abs_diff = (logits_full - logits_chunk).abs().max().item()
    print("Max |Δ| full vs chunk:", max_abs_diff)
    assert torch.allclose(logits_full, logits_chunk, atol=1e-5, rtol=1e-5), \
        f"Mismatch! max abs diff = {max_abs_diff}"


def test_sliding_window_geq_L_matches_full():
    torch.manual_seed(2)
    vocab_size = 509
    d_model = 128
    H = 8
    d_ff = 4 * d_model
    num_layers = 2
    B, L = 2, 7

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0).eval()
    tokens = torch.randint(0, vocab_size, (B, L))

    logits_full = model.forward_train(tokens)

    cache = model.generate_init_cache(B)
    pos = 0
    outs = []
    for t in range(L):
        lt, cache = model.generate_step(tokens[:, t:t+1], cache, pos_offset=pos, window_size=L)  # W >= L
        outs.append(lt); pos += 1
    logits_win = torch.cat(outs, dim=1)

    max_abs_diff = (logits_full - logits_win).abs().max().item()
    print("Max |Δ| full vs window(L):", max_abs_diff)
    assert torch.allclose(logits_full, logits_win, atol=1e-5, rtol=1e-5)


def test_fp16_kv_storage_close_to_full():
    torch.manual_seed(3)
    vocab_size = 521
    d_model = 128
    H = 8
    d_ff = 4 * d_model
    num_layers = 3
    B, L = 2, 10

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0).eval()
    tokens = torch.randint(0, vocab_size, (B, L))

    logits_full = model.forward_train(tokens)

    cache = model.generate_init_cache(B, kv_dtype=torch.float16)
    pos = 0
    outs = []
    for t in range(L):
        lt, cache = model.generate_step(tokens[:, t:t+1], cache, pos_offset=pos, kv_dtype=torch.float16)
        outs.append(lt); pos += 1
    logits_fp16 = torch.cat(outs, dim=1)

    max_abs_diff = (logits_full - logits_fp16).abs().max().item()
    print("Max |Δ| full vs fp16-KV:", max_abs_diff)
    # fp16 rounding may accumulate a bit; keep a slightly looser tolerance
    assert torch.allclose(logits_full, logits_fp16, atol=3e-4, rtol=1e-4), \
        f"Too far with fp16 KV: {max_abs_diff}"


def test_int8_kv_runs_and_is_reasonably_close():
    torch.manual_seed(4)
    vocab_size = 547
    d_model = 128
    H = 8
    d_ff = 4 * d_model
    num_layers = 2
    B, L = 2, 12

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0).eval()
    tokens = torch.randint(0, vocab_size, (B, L))

    logits_full = model.forward_train(tokens)

    cache = model.generate_init_cache(B, kv_int8=True)
    pos = 0
    outs = []
    for t in range(L):
        lt, cache = model.generate_step(tokens[:, t:t+1], cache, pos_offset=pos, kv_int8=True)
        outs.append(lt); pos += 1
    logits_int8 = torch.cat(outs, dim=1)

    max_abs_diff = (logits_full - logits_int8).abs().max().item()
    print("Max |Δ| full vs int8-KV:", max_abs_diff)
    # int8 is approximate; don't assert equality, just sanity-check it's bounded
    assert max_abs_diff < 0.5, f"INT8 deviation too large: {max_abs_diff}"


def test_window_smaller_than_L_keeps_shape_and_grows_correctly():
    torch.manual_seed(5)
    vocab_size = 577
    d_model = 64
    H = 8
    d_ff = 4 * d_model
    num_layers = 2
    B, L, W = 2, 9, 4

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.0).eval()
    cache = model.generate_init_cache(B)
    pos = 0
    for t in range(L):
        tok = torch.randint(0, vocab_size, (B, 1))
        _, cache = model.generate_step(tok, cache, pos_offset=pos, window_size=W)
        pos += 1

    for i, (K, V) in enumerate(cache):
        assert K.shape == (B, H, min(L, W), d_model // H)
        assert V.shape == (B, H, min(L, W), d_model // H)
    print("Windowed cache shapes OK.")


def run_train_and_infer():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 100
    d_model = 32
    H = 4
    num_layers = 2
    d_ff = 4 * d_model
    seq_len = 12
    B = 8

    model = TransformerDecoder(vocab_size, d_model, H, num_layers, d_ff, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dataset = RandomDataset(vocab_size, seq_len, size=200)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

    print("=== Training ===")
    for epoch in range(3):
        print(f"Epoch {epoch+1}")
        train_one_epoch(model, dataloader, optimizer, device)

    print("\n=== Inference ===")
    prompt = torch.randint(0, vocab_size, (B, 5), dtype=torch.long, device=device)
    out = greedy_decode(model, prompt, max_new_tokens=5)
    print("Prompt:\n", prompt)
    print("Generated:\n", out)


if __name__ == "__main__":
    test_training_vs_inference_equivalence()
    test_chunked_vs_full_equivalence()
    test_sliding_window_geq_L_matches_full()
    test_fp16_kv_storage_close_to_full()
    test_int8_kv_runs_and_is_reasonably_close()
    test_window_smaller_than_L_keeps_shape_and_grows_correctly()
    print("✓ All tests passed.")
    run_train_and_infer()
