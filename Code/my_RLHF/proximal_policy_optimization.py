# ./my_RLHF/proximal_policy_optimization.oy


import math, random, os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        self.pad_token_id = self.lm.config.pad_token_id or self.lm.config.eos_token_id
        