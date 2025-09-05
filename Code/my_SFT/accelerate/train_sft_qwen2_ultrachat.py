# my_SFT/raw_DDP/train_sft_qwen2_ultrachat.py

"""
Qwen2-7B + UltraChat-200k SFT baseline with accelerate.

Features:
- TRL SFTTrainer (runs on accelerate)
- Sequence packing (packing=True)
- Response-only label masking (DataCollatorForCompletionOnlyLM)
- LoRA by default (toggle --no_lora for full FT)
- BF16 mixed precision
"""

import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

SYSTEM = "You are a helpful, honest, and concise assistant.\n"

def build_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return (
            f"### System:\n{SYSTEM}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"### System:\n{SYSTEM}\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )
    

def map_ultrachat_row(row):
    """
        Extract the last user → assistant pair as a single-turn SFT example.
        We train using the last pair, because we intend to train the model to answer
        the question instead of continue conversation, the early returns of assitant could 
        be context-dependent, which requires remembering previous turns, and the prompt formatting
        here does not include. Besides, early assiatant answers could be short, incomplete and 
        hallucinated. Given the dataset is already huge, we keep it simple here.
        Each row of UltraChat Datset looks like:
        {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"},
                {"role": "user", "content": "Can you write me a poem about cats?"},
                {"role": "assistant", "content": "Sure! Here's a poem..."}
            ]
        }
    """
    
    if "messages" not in row or not row["messages"]:
        return {"text": None} # no messages return None

    msgs = row["messages"]
    last_a = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
    # find the last assistant meassage
    if last_a is None:
        return {"text": None} # no assitant message return None

    user_idx = None
    for i in range(last_a - 1, -1, -1):
        # scan backward, find the most recent user message
        if msgs[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return {"text": None}

    instr = (msgs[user_idx].get("content") or "").strip()
    out   = (msgs[last_a].get("content") or "").strip()
    if not instr or not out:
        return {"text": None}

    prompt = build_prompt(instruction=instr, input_text="")
    return {"text": prompt + out}


def build_argparser():
    p = argparse.ArgumentParser()
    # core
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B")
    p.add_argument("--dataset", type=str, default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--train_split", type=str, default="train_sft")
    p.add_argument("--eval_split", type=str, default="test_sft")
    p.add_argument("--output_dir", type=str, default="out-qwen2-ultra")
    # training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)  # LoRA-friendly LR
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)  # if > 0, overrides epochs
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    # features & toggles
    p.add_argument("--no_packing", action="store_true")           # default packing=True
    p.add_argument("--no_lora", action="store_true")              # default LoRA on
    p.add_argument("--bf16", action="store_true", default=False)  # toggle via CLI
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    return p

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    # ----- tokenizer & model -----
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    model = 1