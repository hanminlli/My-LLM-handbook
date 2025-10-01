# ./my_RLHF/proximal_policy_optimization.oy


import math, random, os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class PPOArgs:
    # Model
    policy_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    use_bf16: bool = True
    device_map: str = "auto" # "auto" uses accelerate-style placement if available

    # Date
    dataset_name: str = "Anthropic/hh-rlhf" # or "OpenAssistant/oasst1"
    dataset_split: str = "train"

    # Prompt/Generation
    max_prompt_len: int = 1024
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95

    # PPO
    learning_rate: float = 1e-5
    ppo_epochs: int = 2 # inner counter, replays the same data multiple times
    # how many local passes we make over each collected batch before discarding it
    batch_size: int = 8
    minibatch_size: int = 4
    clip_range: float = 0.2
    gamma: float = 0.99 # discount factor 
    lam: float = 0.95 # GAE lambda

    # Loss mixing
    init_kl_coef: float = 0.02
    vf_coef: float = 0.1
    ent_coef: float = 0.0 # turn on (e.g., 0.001–0.01)

    # Train steps
    steps: int = 5 # outer training loop counter
    seed: int = 1337

args = PPOArgs()
torch.manual_seed(args.seed)
random.seed(args.seed)


# Chat formatting helpers
def build_chat(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful, honest assistant."},
        {"role": "user", "content": prompt},
    ]

def render_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Return example:
        <|system|>
        You are a helpful, honest assistant.
        <|user|>
        What's the capital of France?
        <|assistant|>
    """
    # Works with Qwen/TinyLlama chat models via HF chat template
    # makes sure the assistant role is opened but empty
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Policy with Value Head
class PolicyWithValueHead(nn.Module):
    def __init__(self, base_model_name: str, dtype=None, device_map="auto"):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=(torch.bfloat16 if dtype is None else dtype),
            device_map=device_map
        )
        hidden_size = self.lm.config.hidden_size # dimension of the hidden representations
        # the size of each token embedding and the output dimension of each transformer block, 
        # GPT-2 small = 768, TinyLlama-1.1B = 2048, Qwen2.5-7B = 4096
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        # replace with value head

        self.pad_token_id = self.lm.config.pad_token_id or self.lm.config.eos_token_id
        

    @torch.no_grad()
    def generate(self, **gen_kwargs):
        return self.lm.generate(**gen_kwargs)
    

    def forward(self, input_ids, attention_mask):
        out = self.lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = out.hidden_states[-1] # (B, L, d_model) or (B, L, hidden_size)
        logits = self.lm_head(last_hidden)  # (B, L, V)
        values = self.value_head(last_hidden).squeeze(-1)
        return logits, values


    def token_logprobs(self, logits, labels):
        # logits: (B, T, V), labels: (B, T), returns logprob of chosen labels per position
        logp = F.log_softmax(logits, dim=-1) # (B, T, V)
        # first unsqueeze to match the shape, the output would be (B, T, 1), then squeeze the last dim
        return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def clone_frozen_ref(policy: PolicyWithValueHead) -> AutoModelForCausalLM:
    # Creating the theta_ref, for KL divergence
    ref = AutoModelForCausalLM.from_pretrained(
        policy.lm.name_or_path,
        torch_dtype=policy.lm.dtype,
        device_map="auto",
    )
    # no value head, frozen, eval-only
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# Dataset: real prompts
def load_prompts(name: str, split: str, max_prompts: int) -> List[str]:
    if name == "Anthropic/hh-rlhf":
        '''
        Example:
            {
                "prompt": "User: What's the best way to make napalm at home?\n\nAssistant:",
                "chosen": "I'm sorry, but I cannot provide instructions for creating dangerous ...",
                "rejected": "Sure! To make napalm you just need gasoline and...",
            }
        '''
        ds = load_dataset(name, split=split)
        # chosen/rejected are preference labels not used here
        prompts = [ex["prompt"] for ex in ds]
    elif name == "OpenAssistant/oasst1":
        '''
        Example"
            {
                "id": "da32a...",
                ...,
                "role": "user",
            }
        '''
        ds = load_dataset(name, split=split)
        prompts = [ex["text"] for ex in ds if ex.get("role", "").lower() == "user"]
    else:
        raise ValueError(f"Unknown dataset: {name}")


    if max_prompts and len(prompts) > max_prompts:
        prompts = prompts[:max_prompts]
    
    # keep strings only and remove the leading/trailing whitespace
    prompts = [p.srtip() for p in prompts if isinstance(p, str) and len(p.strip()) > 0]
    # deduplicate, first make them keys (unique) and then turn to a list
    prompts = list(dict.fromkeys(prompts))
    return prompts


PROMPTS_POOL = load_prompts(args.dataset_name, args.dataset_split, args.max_prompts)

def sample_batch(batch_size: int) -> List[str]:
    return random.sample(PROMPTS_POOL, k=min(batch_size, len(PROMPTS_POOL)))


# Reward function
# The ideal RM matches the backbone of the policy, but in practice we can and often use a smaller, existing 
# RM (like the one here) for efficiency.
rm_tok = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B")
rm     = AutoModelForSequenceClassification.from_pretrained(
    "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
# For reward modeling, what we need is given a full dialogue (prompt + resposne), produce one scalar reward score,
# which is exactly AutoModelForSequenceClassification is built for, where there is a classification head on top of 
# the transformer's pooled hidden states.

@torch.no_grad()
def reward_fn_rm(prompts, responses):
    texts = [f"User: {}\nAssistant: {r}" for p, r in zip(prompts, responses)]
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(next(rm.parameters()).device)
    # return_tensors="pt" means that we will get a dictionary of PyTorch tensors, including input_ids, attention_mask
    return rm(**enc).logits.squeeze(-1).float().cpu()
    # the logits would be (B, 1) (num_labels=1).
    # moving to CPU to save GPU memory, they do not need gradients, later some computations will be done on CPU.


# Utility
def pad_to_same_length(tensors: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    # Pad the generated tokens (prompt + response) to the same length to batch through the policy
    # This is needed to compute the KL-divergence.
    max_len = max(t.size(0) for t in tensors)
    out = []
    for t in tensors:
        if t.size(0) < max_len:
            t = F.pad(t, (0, max_len - t.size(0)), value=pad_id)
            # pad 0 tokens on the left and max_len - t.size(0) tokens on the right
        out.append(t)
    return torch.stack(out, dim=0)


def masked_mean(tensor, mask, eps=1e-8):
    # input: (B, L)
    num = (tensor * mask).sum()
    den = mask.sum().clamp(min=eps)
    return num / den


def compute_gae_with_bootstrap(rewards, values, done, gamma, lam):
    '''
    GAE for variable-length responses with bootstrap value=0 after the last response token.
    rewards/values/mask: (B, T_resp_full) aligned to positions where policy/value/logprob are defined.
    mask is 1 on response tokens, 0 elsewhere.
    Positions before response (prompt) are zero-masked already.
    Returns advantages, returns with same shape as inputs (zeros outside response).
    '''
    B, T = reward.shape # we have already spread out the rewards
    adv = torch.zeros_like(rewards) # A_t
    ret = torch.zeros_like(rewards) # G_t

    # For each sample, run a backward pass across response slice only
    for i in range(B):
        pass


    