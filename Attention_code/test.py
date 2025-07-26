import torch
from my_attention.multi_head_self_attention import MultiHeadSelfAttention
from my_attention.multi_query_self_attention import MultiQuerySelfAttention 
from my_attention.grouped_query_attention import GroupedQueryAttention

class Test(torch.nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.MHA = MultiHeadSelfAttention(d_model=512, num_heads=8, use_numrical_softmax=True)
        self.MQA = MultiQuerySelfAttention(d_model=512, num_heads=8)
        self.GQA = GroupedQueryAttention(d_model=512, num_q_heads=32, num_kv_heads=4)
    
    def forward(self, x):
        return self.GQA(x)


def main():
    model = Test()
    x = torch.randn(1, 10, 512)
    output = model(x)
    print(output.shape)

if __name__ == "__main__":
    main()






