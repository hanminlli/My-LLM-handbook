import torch
import torch.nn as nn

# ---- ATTENTION MODULES ----
from my_attention.multi_head_self_attention import MultiHeadSelfAttention
from my_attention.multi_query_self_attention import MultiQuerySelfAttention
from my_attention.grouped_query_attention import GroupedQueryAttention
from my_attention.multi_head_latent_attention import MultiHeadLatentAttention
from my_attention.multi_head_latent_attention import LatentTransformer
from my_normlization.rms_norm import RMSNorm
from my_normlization.layer_norm import LayerNorm
from my_attention.encoder import MLP, GatedMLP
from my_attention.parallel_attention import ParallelTransformerBlock

# ---- TEST MODULE FOR ATTENTION ----
class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.MHA = MultiHeadSelfAttention(d_model=512, num_heads=8, use_numrical_softmax=True)
        self.MQA = MultiQuerySelfAttention(d_model=512, num_heads=8)
        self.GQA = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=4)
        self.MLA = MultiHeadLatentAttention(d_model=512, num_latents=5, num_heads=8)

    def forward(self, x):
        return {
            "MHA": self.MHA(x),             # (B, L, 512)
            "MQA": self.MQA(x),             # (B, L, 512)
            "GQA": self.GQA(x),             # (B, L, 512)
            "MLA": self.MLA(x),             # (B, 5, 512)
        }

def test_latent_transformer():
    x = torch.randn(1, 10, 512)  # dummy input: (batch=1, seq_len=10, dim=512)
    model = Test()
    out_dict = model(x)
    print("Original Test outputs:")
    for k, v in out_dict.items():
        print(f"  {k:>3} -> {tuple(v.shape)}")

    latent_enc = LatentTransformer(d_model=512, num_latents=5, num_heads=8, num_layers=2)
    z_out = latent_enc(x)
    print("\nLatentTransformer output:", tuple(z_out.shape))   # expected (1, 5, 512)

def test_attention_shapes_and_validity():
    x = torch.randn(2, 7, 512)  # (batch=2, seq_len=7, dim=512)
    model = Test()
    out_dict = model(x)

    # Check output shapes
    assert out_dict["MHA"].shape == (2, 7, 512), f"MHA shape wrong: {out_dict['MHA'].shape}"
    assert out_dict["MQA"].shape == (2, 7, 512), f"MQA shape wrong: {out_dict['MQA'].shape}"
    assert out_dict["GQA"].shape == (2, 7, 512), f"GQA shape wrong: {out_dict['GQA'].shape}"
    assert out_dict["MLA"].shape == (2, 5, 512), f"MLA shape wrong: {out_dict['MLA'].shape}"

    # Check for NaN/Inf values
    for name, v in out_dict.items():
        assert torch.isfinite(v).all(), f"{name} output contains NaN or Inf"

    latent_enc = LatentTransformer(d_model=512, num_latents=5, num_heads=8, num_layers=2)
    z_out = latent_enc(x)
    assert z_out.shape == (2, 5, 512), f"LatentTransformer shape wrong: {z_out.shape}"
    assert torch.isfinite(z_out).all(), "LatentTransformer output contains NaN or Inf"

    print("All attention tests passed!")

def test_rmsnorm_basic():
    B, L, D = 3, 4, 5
    x = torch.randn(B, L, D)
    norm = RMSNorm(d_model=D)
    y = norm(x)
    
    # Check output shape
    assert y.shape == (B, L, D), f"RMSNorm output shape incorrect: got {y.shape}, expected {(B, L, D)}"
    # Check for NaN or Inf
    assert torch.isfinite(y).all(), "RMSNorm output contains NaN or Inf"
    # Check that scale is learnable
    assert norm.scale.requires_grad, "scale parameter is not learnable"
    
    print("RMSNorm basic test passed.")

def test_layernorm_basic():
    B, L, D = 3, 4, 5
    x = torch.randn(B, L, D)
    norm = LayerNorm(d_model=D)
    y = norm(x)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()
    print("LayerNorm basic test passed.")

def test_mlp_basic():
    B, L, D = 3, 4, 5
    x = torch.randn(B, L, D)
    mlp = MLP(d_model=D, d_ffn=10, activation="gelu_tanh")
    y = mlp(x)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()
    print("MLP basic test passed.")

def test_gated_mlp_basic():
    B, L, D = 3, 4, 5
    x = torch.randn(B, L, D)
    mlp = GatedMLP(d_model=D, d_ffn=10, gate_activation="silu")
    y = mlp(x)
    assert y.shape == (B, L, D)
    assert torch.isfinite(y).all()
    print("GatedMLP basic test passed.")


def test_parallel_attention():
    torch.manual_seed(0)

    B, T, d_model = 2, 6, 32
    num_heads = 4
    d_ffn = 64
    dropout = 0.1

    block = ParallelTransformerBlock(
        d_model=d_model, num_heads=num_heads, d_ffn=d_ffn, dropout=dropout
    )

    x = torch.randn(B, T, d_model)

    y = block(x)
    print("[shape]", y.shape)
    print("[finite]", torch.isfinite(y).all().item())

    y.sum().backward()
    total_grad = 0.0
    num_params = 0
    with torch.no_grad():
        for p in block.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_grad += g.abs().sum().item()
                num_params += 1
    print("[params_with_grad]", num_params)
    print("[total_grad_abs_sum]", round(total_grad, 4))

    block.train()
    y1 = block(x)
    y2 = block(x)
    print("[train_equal]", torch.allclose(y1, y2))

    block.eval()
    y3 = block(x)
    y4 = block(x)
    print("[eval_equal]", torch.allclose(y3, y4))

    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    block.eval()
    y_nomask = block(x)
    y_mask = block(x, mask=causal_mask)
    diff = (y_nomask - y_mask).abs().mean().item()
    print("[mask_effect_mean_abs_diff]", round(diff, 6))

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 6:  # Pascal+ typically OK for cu118+
                device_block = ParallelTransformerBlock(d_model, num_heads, d_ffn, dropout).cuda()
                x_cuda = x.cuda()
                y_cuda = device_block(x_cuda)
                print("[cuda]", y_cuda.is_cuda, y_cuda.shape)
            else:
                print(f"[cuda] Skipping: compute capability {major}.{minor} not supported by this torch build.")
        except Exception as e:
            print("[cuda] Skipping due to error:", e)


if __name__ == "__main__":
    test_parallel_attention()