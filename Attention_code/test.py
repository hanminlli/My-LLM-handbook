import torch
from my_attention.multi_head_self_attention import MultiHeadSelfAttention
from my_attention.multi_query_self_attention import MultiQuerySelfAttention
from my_attention.grouped_query_attention import GroupedQueryAttention
from my_attention.multi_head_latent_attention import MultiHeadLatentAttention
from my_attention.multi_head_latent_attention import LatentTransformer


class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.MHA = MultiHeadSelfAttention(d_model=512, num_heads=8, use_numrical_softmax=True)
        self.MQA = MultiQuerySelfAttention(d_model=512, num_heads=8)
        self.GQA = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=4)
        self.MLA = MultiHeadLatentAttention(d_model=512, num_latents=5, num_heads=8)

    def forward(self, x):
        # return whichever attention mechanism you want to inspect
        return {
            "MHA": self.MHA(x),             # (B, L, 512)
            "MQA": self.MQA(x),             # (B, L, 512)
            "GQA": self.GQA(x),             # (B, L, 512)
            "MLA": self.MLA(x),             # (B, 5, 512)
        }
    
def main():
    x = torch.randn(1, 10, 512)  # dummy input: (batch=1, seq_len=10, dim=512)

    # test original wrapper
    model = Test()
    out_dict = model(x)
    print("Original Test outputs:")
    for k, v in out_dict.items():
        print(f"  {k:>3} -> {tuple(v.shape)}")

    # test stacked latent transformer
    latent_enc = LatentTransformer(d_model=512, num_latents=5, num_heads=8, num_layers=2)
    z_out = latent_enc(x)
    print("\nLatentTransformer output:", tuple(z_out.shape))   # expected (1, 5, 512)

if __name__ == "__main__":
    main()