# ./my_RLHF/reward_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

class RewardModel(nn.Module):

    def __init__(self, base_model_name: str, pooling: str = "last", dtype=None, device_map="auto"):
        """
            Reward model built on top of a pretrained LM backbone.
            - base_model_name: e.g. "Qwen/Qwen2.5-7B-Instruct"
            - pooling: "last" = use last token hidden state
                    "mean" = mean pool over tokens
                    "cls"  = use [CLS] token if model supports it
        """
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=(torch.bfloat16 if dtype is None else dtype),
            device_map=device_map,
            output_hidden_states=True
        )
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)
        self.pooling = pooling


    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = out.last_hidden_state # (B, L, d_model)

        # Pooling to get sequence representation
        if self.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1 # index of the last nonpad
            pooled = last_hidden[torch.arange(last_hidden.size(0)), lengths]
            # from (B, L, d_model) to (B, d_model)
        elif self.pooling == "mean":
            pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
            # (B, L, d_model) * (B, L, 1) / (B, L, 1)
        elif self.pooling == "cls" and self.backbone.config.model_type == "bert":
            pooled = last_hidden[:, 0] # [CLS] embedding
        else:
            raise ValueError(f"Unsupported pooling={self.pooling}")

        reward = self.reward_head(pooled).squeeze(-1)  # (B,)
        return reward


if __name__ == "__main__":
    base_model = "bert-base-uncased"  # test
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    
    # Anthropic HH RLHF: preference pairs
    ds = load_dataset("Anthropic/hh-rlhf", split="train[:2000]")  # test

    model = RewardModel(base_model).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for step, ex in enumerate(ds):
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        # Pack prompt + response
        text_chosen = f"User: {prompt}\nAssistant: {chosen}"
        text_rejected = f"User: {prompt}\nAssistant: {rejected}"

        # Tokenize
        enc_chosen = tok(text_chosen, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc_rejected = tok(text_rejected, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Forward pass
        r_chosen = model(enc_chosen["input_ids"], enc_chosen["attention_mask"])
        r_rejected = model(enc_rejected["input_ids"], enc_rejected["attention_mask"])

        # Bradley–Terry / logistic loss
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step} | Loss {loss.item():.4f} | r_chosen {r_chosen.item():.3f}, r_rejected {r_rejected.item():.3f}")

        if step > 500:  # stop early for demo
            break