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

def main():
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


if __name__ == "__main__":
    test_mlp_basic()
    test_gated_mlp_basic()