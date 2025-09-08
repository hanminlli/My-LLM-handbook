# my_attention/LoRA.py

"""
Tiny GPT-2 style model (2 layers) with LoRA and minimal QLoRA.

Run examples:
    # Plain training (no adapters)
    python tiny_gpt2_lora_qlo.py --mode none

    # Manual LoRA on attn.q & attn.v
    python tiny_gpt2_lora_qlo.py --mode lora

    # Minimal QLoRA (4-bit symmetric per-row quantization) on attn.q & attn.v
    python tiny_gpt2_lora_qlo.py --mode qlora

    # Expand targets
    python tiny_gpt2_lora_qlo.py --mode lora --targets attn.q, attn.v, attn.out, mlp.fc1, mlp.fc2

Options:
    --amp true      # try autocast mixed precision
    --steps 100     # training steps
    --lr 5e-4       # learning rate
"""

import math
import argparse
from dataclasses import dataclass
from typing import List, Optional
from torch.amp import GradScaler, autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    bnb = None
    _HAS_BNB = False


class LoRALinear(nn.Module):
    """
    Manual LoRA around a frozen base Linear.
    """

    def __init__(self, d_in, d_out, bias=True, r=8, alpha=16.0, lora_dropout=0.0):
        super().__init__()
        self.base = nn.Linear(d_in, d_out, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, d_in))
            self.lora_B = nn.Parameter(torch.zeros(d_out, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialized like a normal linear layer, in this case the projection has reasonable variance 
            # and this ensures that activation do not collapse.
            nn.init.zeros_(self.lora_B)
            # We start as all zeros so that at the very start of the training, LoRA contributes nothing and the
            # model behaves exactly like the pretrained model.
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            # every parameter we want to track must be registered, 
            # setting it to none to highlight it's currently none
        
        self.merged = False

    def forward(self, x):
        # x: (B, L, d_in)
        out = F.linear(x, self.base.weight, self.base.bias)
        if self.r > 0:
            A = self.lora_A.to(x.dtype)
            B = self.lora_B.to(x.dtype)
            xA = self.lora_dropout(x) @ A.t() # (B, L, d_in) @ (r, d_in)^T
            xAB = xA @ B.t() # (B, L, r) @ (d_out, r)^T
            out = out + self.scaling * xAB
        return out
    
    @torch.no_grad()
    def merge_lora(self):
        """
        Merge LoRA delta into base.weight for inference.
        """
        if self.merged or self.r == 0:
            return
        deltaW = (self.lora_B @ self.lora_A) * self.scaling
        # Notice that in PyTorch we store matrix W but when its applied in linear layers 
        # we do something like x W^T instead of Wx
        self.base.weight.add_(deltaW)
        self.merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        '''
        Revert a previous merge.
        '''
        if not self.merged or self.r == 0:
            return
        deltaW = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.sub_(deltaW)
        self.merged = False


class ToySymm4bitQLoRALinear(nn.Module):
    """
    Minimal educational QLoRA linear with int4
    The absence of merge option is by design, since merge requries dequantize and then merge, which is lossy
    , and requires larger storage, we leave it as always computed on the fly. 
    """

    def __init__(self, d_in, d_out, bias=True, r=8, alpha=16.0, lora_dropout=0.0, dtype=torch.float32):
        super().__init__()
        base = nn.Linear(d_in, d_out, bias=bias)

        # First quantization
        with torch.no_grad():
            W = base.weight.detach().float()
            # We temporarily lift the weights into float32 for safer math,
            # then later re-cast into the working dtype.
            max_abs = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            scale   = (max_abs / 7.0) # map to int8 in [-8, 7]
            qweight = torch.round((W / scale)).clamp(-8, 7).to(torch.int8)

        self.register_buffer("qweight", qweight, persistent=True) # not trainable, but move to the right place when call .cuda()
        # persistent=True indicates included in state dict, saved in checkpoints

        # Second quantization
        with torch.no_grad():
            s_absmax   = scale.abs().amax()  # scalar
            meta_scale = (s_absmax / 127.0)
            qscale     = torch.round(scale / meta_scale).clamp(-127, 127).to(torch.int8)

        self.register_buffer("qscale", qscale, persistent=True)
        self.register_buffer("meta_scale", meta_scale.to(dtype), persistent=True)

        if bias:
            self.bias = nn.Parameter(base.bias.data.to(dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0 
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.dtype = dtype

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, d_in, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
        

    def forward(self, x):
        scale_row = (self.qscale.to(torch.float32) * self.meta_scale) # keep fp32
        W         = (self.qweight.to(torch.float32) * scale_row).to(x.dtype) # back to x's dtype
        out = F.linear(x, W, self.bias)
        if self.r > 0:
            A = self.lora_A.to(x.dtype)
            B = self.lora_B.to(x.dtype)
            xA = self.lora_dropout(x) @ A.t()
            xAB = xA @ B.t()
            out = out + self.scaling * xAB
        return out


class NF4QLoRALinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True, r=8, alpha=16.0, lora_dropout=0.0, 
                 compute_dtype=torch.bfloat16):
        super().__init__()
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available for NF4QLoRALinear")
        
        self.base = bnb.nn.Linear4bit(
            d_in,
            d_out,
            bias=bias,
            compute_dtype=compute_dtype,
            compress_statistics=True, # double quantization
            quant_type="nf4",
        )

        # Freeze all base params
        for p in self.base.parameters():
            p.requires_grad = False
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, d_in, dtype=compute_dtype))
            self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=compute_dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        
    def forward(self, x):
        out = self.base(x)  # bnb handles dequantization internally
        if self.r > 0:
            A = self.lora_A.to(x.dtype)
            B = self.lora_B.to(x.dtype)
            xA = self.lora_dropout(x) @ A.t()
            xAB = xA @ B.t()
            out = out + self.scaling * xAB
        return out
        

# Tiny GPT-2 style components
@dataclass
class TinyConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    d_ff: int = 512
    max_seq_len: int = 64
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    layer_norm_eps: float = 1e-5
    # LoRA-related
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # Which modules get LoRA: "attn.q", "attn.v", "attn.out", "mlp.fc1", "mlp.fc2"
    lora_targets: Optional[List[str]] = None
    # mode: "none", "lora", "qlora"
    mode: str = "lora"


class CausalSelfAttention(nn.Module):

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.d_head= cfg.d_model // cfg.n_head

        Linear = self._linear_factory
        self.q = Linear(cfg.d_model, cfg.d_model, name="attn.q")
        self.k = nn.Linear(cfg.d_model, cfg.d_model, bias=True) # usually not adapted
        self.v = Linear(cfg.d_model, cfg.d_model, name="attn.v")
        self.out = Linear(cfg.d_model, cfg.d_model, name="attn.out")

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool)).view(1,1,cfg.max_seq_len,cfg.max_seq_len)
        # take the lower triangular including diagonal
        self.register_buffer("causal_mask", mask, persistent=False)


    
    def _linear_factory(self, d_in, d_out, name: str):
        cfg = self.cfg
        use_lora = (cfg.mode in ("lora", "qlora")) and (cfg.lora_targets and name in cfg.lora_targets)

        if not use_lora:
            return nn.Linear(d_in, d_out, bias=True) 
        if cfg.mode == "lora":
            return LoRALinear(d_in, d_out, bias=True, r=cfg.lora_r, 
                              alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        else:
            # via bitsandbytes if available, fallback to toy symmetric 4-bit (with double-quantized scales)
            if _HAS_BNB:
                return NF4QLoRALinear(d_in, d_out, bias=True, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                      lora_dropout=cfg.lora_dropout, compute_dtype=torch.bfloat16)
            else:
                print("[warn] bitsandbytes not found, falling back to Toy 4-bit (double-quantized scales) for", name)
                return ToySymm4bitQLoRALinear(d_in, d_out, bias=True, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                               lora_dropout=cfg.lora_dropout, dtype=torch.float32)
            
    
    def forward(self, x):
        B, T, d_model = x.shape
        q = self.q(x).view(B, T, self.n_head, self.d_head).transpose(1, 2) # (B, H, T, d_head)
        k = self.k(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))
        # masked_fill fill where there is True, so we invert
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, d_model)
        y = self.resid_drop(self.out(y))
        return y


class MLP(nn.Module):

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        Linear = self._linear_factory
        self.fc1 = Linear(cfg.d_model, cfg.d_ff, name="mlp.fc1")
        self.fc2 = Linear(cfg.d_ff, cfg.d_model, name="mlp.fc2")
        self.act = nn.GELU()

    
    def _linear_factory(self, in_f, out_f, name: str):
        cfg = self.cfg
        use_lora = (cfg.mode in ("lora", "qlora")) and (cfg.lora_targets and name in cfg.lora_targets)
        if not use_lora:
            return nn.Linear(in_f, out_f, bias=True)
        if cfg.mode == "lora":
            return LoRALinear(in_f, out_f, bias=True, r=cfg.lora_r, alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        else:
            if _HAS_BNB:
                return NF4QLoRALinear(in_f, out_f, bias=True, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                      lora_dropout=cfg.lora_dropout, compute_dtype=torch.bfloat16)
            else:
                print("[warn] bitsandbytes not found, falling back to Toy 4-bit (double-quantized scales) for", name)
                return ToySymm4bitQLoRALinear(in_f, out_f, bias=True, r=cfg.lora_r, alpha=cfg.lora_alpha,
                                               lora_dropout=cfg.lora_dropout, dtype=torch.float32)
        
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))



class Block(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT2(nn.Module):

    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        # trainable positional embeddings, modern LLMs do not use it anymore, usually turn to a predefined function
        self.drop = nn.Dropout(cfg.resid_dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights
        self.apply(self._init_weights)
        self.lm_head.weight = self.tok_emb.weight
        
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long).unsqueeze(0)
        # tok_emb, pos_emb give (B, T, d_model) 
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    

# Helper functions
def freeze_non_lora_params(model: nn.Module):
    """
    Freeze everything, unfreeze only LoRA A/B and LayerNorms.
    (When mode == qlora with bnb, base 4-bit weights are already non-trainable.)
    """
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, (LoRALinear, NF4QLoRALinear, ToySymm4bitQLoRALinear)) and getattr(module, "r", 0) > 0:
            if hasattr(module, "lora_A") and module.lora_A is not None:
                module.lora_A.requires_grad_(True)
                # requires_grad = True works in most cases, but in case of already moved models between devices
                # sometimes its not clean so we use requires_grad_(True) to ensure internal state updated consistently
            if hasattr(module, "lora_B") and module.lora_B is not None:
                module.lora_B.requires_grad_(True)
        if isinstance(module, nn.LayerNorm):
            # they control scale and shift of activations at every layer
            # allows model to better adapt
            for p in module.parameters():
                p.requires_grad = True


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def approx_model_bytes(model: torch.nn.Module) -> int:
    """
    Very rough memory accounting:
        LoRA: base fp weights + (A/B) adapters.
        NF4/Linear4bit: 4-bit weights (~0.5 byte/elt) + (compressed) stats + bias + A/B.
        Toy QLoRA: int4 weights + int8 qscale per row + one fp meta_scale + bias + A/B.
    We avoid double counting by tracking parameter IDs already included.
    """
    def dtype_size(dt: torch.dtype) -> int:
        if dt == torch.float32:
            return 4
        if dt in (torch.float16, torch.bfloat16):
            return 2
        if dt == torch.int8:
            return 1
        return 4

    bytes_total = 0
    accounted_param_ids = set()

    for m in model.modules():
        if isinstance(m, NF4QLoRALinear) and _HAS_BNB:
            d_in = m.base.in_features
            d_out = m.base.out_features

            # 4-bit main weight
            q_bytes = int(d_in * d_out * 0.5)

            # bias 
            bias_bytes = 0
            if m.base.bias is not None:
                bias_bytes = m.base.bias.numel() * dtype_size(m.base.bias.dtype)
                accounted_param_ids.add(id(m.base.bias))

            # LoRA adapters
            ab_bytes = 0
            if m.r > 0:
                ab_bytes += m.lora_A.numel() * dtype_size(m.lora_A.dtype)
                ab_bytes += m.lora_B.numel() * dtype_size(m.lora_B.dtype)
                accounted_param_ids.add(id(m.lora_A))
                accounted_param_ids.add(id(m.lora_B))

            bytes_total += q_bytes + bias_bytes + ab_bytes

        elif isinstance(m, ToySymm4bitQLoRALinear):
            q_bytes = int(m.qweight.numel() * 0.5)   # int4
            qscale_bytes = m.qscale.numel() * 1      # int8 per row
            meta_bytes = 4                            # one fp32 scalar

            bias_bytes = 0
            if m.bias is not None:
                bias_bytes = m.bias.numel() * dtype_size(m.bias.dtype)
                accounted_param_ids.add(id(m.bias))

            ab_bytes = 0
            if m.r > 0:
                ab_bytes += m.lora_A.numel() * dtype_size(m.lora_A.dtype)
                ab_bytes += m.lora_B.numel() * dtype_size(m.lora_B.dtype)
                accounted_param_ids.add(id(m.lora_A))
                accounted_param_ids.add(id(m.lora_B))

            bytes_total += q_bytes + qscale_bytes + meta_bytes + bias_bytes + ab_bytes

        elif isinstance(m, LoRALinear):
            # Base fp weights/bias 
            base_w = m.base.weight
            bytes_total += base_w.numel() * dtype_size(base_w.dtype)
            accounted_param_ids.add(id(base_w))

            if m.base.bias is not None:
                base_b = m.base.bias
                bytes_total += base_b.numel() * dtype_size(base_b.dtype)
                accounted_param_ids.add(id(base_b))

            # LoRA A/B 
            if m.r > 0:
                bytes_total += m.lora_A.numel() * dtype_size(m.lora_A.dtype)
                bytes_total += m.lora_B.numel() * dtype_size(m.lora_B.dtype)
                accounted_param_ids.add(id(m.lora_A))
                accounted_param_ids.add(id(m.lora_B))

    # Add everything else 
    for p in model.parameters():
        if id(p) in accounted_param_ids:
            continue
        bytes_total += p.numel() * (2 if p.dtype in (torch.float16, torch.bfloat16) else 4)

    return bytes_total


def toy_data(vocab_size, batch_size, seq_len, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.roll(shifts=-1, dims=1)  # predict next token (last token into front)
    return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["none", "lora", "qlora"], default="lora",
                        help="Which variant to use for targeted modules.")
    parser.add_argument("--targets", type=str, default="attn.q,attn.v",
                        help="Comma-separated module names to apply (Q)LoRA to: e.g., attn.q,attn.v,attn.out,mlp.fc1,mlp.fc2")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", type=str, default="false", help="true/false to enable autocast")
    args = parser.parse_args()

    amp = args.amp.lower() in ("1", "true", "yes", "y")

    cfg = TinyConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        mode=args.mode
    )

    device = torch.device(args.device)
    torch.manual_seed(42)

    model = TinyGPT2(cfg).to(device)

    if cfg.mode in ("lora", "qlora"):
        freeze_non_lora_params(model)

    
     # Show whether real NF4 is active
    print(f"Mode: {cfg.mode}")
    print(f"Targets: {cfg.lora_targets}")
    if cfg.mode == "qlora":
        print("Real NF4:", _HAS_BNB)

    total, trainable = count_params(model)
    bytes_est = approx_model_bytes(model)
    print(f"Total params: {total:,} | Trainable: {trainable:,} | Approx bytes: ~{bytes_est/1e6:.2f} MB")

    # Optimizer over trainable params only
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # automatic mixed precision，dynamically scales up the loss before backward pass
    USE_BNB_RUNTIME = _HAS_BNB and device.type == "cuda"
    using_nf4 = (cfg.mode == "qlora" and USE_BNB_RUNTIME)
    amp_dtype = torch.bfloat16 if using_nf4 else torch.float16
    scaler = GradScaler('cuda', enabled=amp and device.type == 'cuda' and amp_dtype == torch.float16)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = toy_data(cfg.vocab_size, args.batch_size, cfg.max_seq_len, device)
        with autocast('cuda', enabled=amp and device.type == 'cuda', dtype=amp_dtype):
            logits = model(x)  # [B, T, V]
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

        if step % 10 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f}")
        
    if cfg.mode == "lora":
        merged = 0
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.merge_lora()
                merged += 1
        if merged:
            print(f"Merged {merged} LoRA modules into base weights for inference.")


if __name__ == "__main__":
    main()
