# my_SFT/raw_DDP/train_sft_qwen2_ultrachat_wandb.py

"""
Qwen2-7B + UltraChat-200k SFT baseline with accelerate + W&B.

Changes vs your original:
- Uses a TrainerCallback to compute & log perplexity from eval_loss (compute_metrics isn't called with eval_loss).
- Safer gradient checkpointing (use_reentrant=False when available).
- Ensures model.config.pad_token_id matches tokenizer.
- Drops extra dataset columns (keeps only "text").
- Optional EOS append to targets.
- W&B: supports --run_name and --wandb_project (or env WANDB_PROJECT).
"""

import argparse
import math
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, TrainerCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, PeftModel

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
    See original comments for rationale. Returns a dict with a single key "text".
    """
    if "messages" not in row or not row["messages"]:
        return {"text": None}

    msgs = row["messages"]
    last_a = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
    if last_a is None:
        return {"text": None}

    user_idx = None
    for i in range(last_a - 1, -1, -1):
        if msgs[i].get("role") == "user":
            user_idx = i
            break
    if user_idx is None:
        return {"text": None}

    instr = (msgs[user_idx].get("content") or "").strip()
    out = (msgs[last_a].get("content") or "").strip()
    if not instr or not out:
        return {"text": None}

    prompt = build_prompt(instruction=instr, input_text="")
    return {"text": prompt + out}


class PerplexityCallback(TrainerCallback):
    """Compute eval perplexity from eval_loss and inject into metrics & stdout."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float("inf")
            metrics["eval_ppl"] = ppl
            print(f"[eval] step={state.global_step} loss={metrics['eval_loss']:.4f} ppl={ppl:.3f}")


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
    p.add_argument("--save_total_limit", type=int, default=3, help="Keep at most this many checkpoints.")
    # wandb
    p.add_argument("--run_name", type=str, default="qwen2-ultra-sft")
    p.add_argument("--wandb_project", type=str, default=None, help="Overrides env WANDB_PROJECT if set.")
    p.add_argument("--append_eos", action="store_true", help="Append EOS to each sample text.")
    p.add_argument("--save_merged", action="store_true", help="After training, merge LoRA adapters into the base model and save a standalone checkpoint.")
    p.add_argument("--merged_dir", type=str, default=None, help="Optional explicit path for merged checkpoint (defaults to output_dir + '-merged').")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    # ----- W&B env (optional) -----
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_WATCH", "false")  # avoid excessive logging of gradients/parameters

    # ----- tokenizer & model -----
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map=None,  # Accelerate decides on placement
        attn_implementation="sdpa",  # safe default on A100 with bf16
    )

    # Ensure model config matches tokenizer for padding, and disable KV cache during FT
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # older transformers fallback
            model.gradient_checkpointing_enable()

    # Only needed if you add new tokens. With untouched vocab this is a no-op, but harmless.
    model.resize_token_embeddings(len(tok))

    # ----- datasets -----
    ds_train = load_dataset(args.dataset, split=args.train_split)
    ds_eval = load_dataset(args.dataset, split=args.eval_split)

    ds_train = ds_train.map(
        map_ultrachat_row,
        remove_columns=[c for c in ds_train.column_names if c != "messages"],
        num_proc=8,
    )

    ds_eval = ds_eval.map(
        map_ultrachat_row,
        remove_columns=[c for c in ds_eval.column_names if c != "messages"],
        num_proc=8,
    )

    ds_train = ds_train.filter(lambda r: r["text"] is not None and len(r["text"]) > 0)
    ds_eval = ds_eval.filter(lambda r: r["text"] is not None and len(r["text"]) > 0)

    # keep only the training text column to save memory
    ds_train = ds_train.remove_columns([c for c in ds_train.column_names if c != "text"])
    ds_eval = ds_eval.remove_columns([c for c in ds_eval.column_names if c != "text"])

    # (optional) append EOS so the model learns an explicit stopping token
    if args.append_eos:
        ds_train = ds_train.map(lambda r: {"text": r["text"] + tok.eos_token})
        ds_eval = ds_eval.map(lambda r: {"text": r["text"] + tok.eos_token})

    # ----- Response-only masking -----
    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Response:\n",
        tokenizer=tok,
    )

    # ----- LoRA -----
    peft_cfg = None
    if not args.no_lora:
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,  # effective scale alpha/r = 2
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    # ----- Trainer -----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        peft_config=peft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_eval.select(range(min(2000, len(ds_eval)))),  # fast sanity eval
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=not args.no_packing,
        data_collator=collator,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        num_train_epochs=None if args.max_steps > 0 else args.num_train_epochs,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        report_to=["wandb"],
        run_name=args.run_name,
        save_safetensors=True,
        callbacks=[PerplexityCallback()],
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("Training finished. Weights saved to:", args.output_dir)

    # Merge LoRA adapters into a standalone checkpoint
    if args.save_merged:
        merged_dir = args.merged_dir or (args.output_dir + "-merged")
        try:
            model_for_save = trainer.model
            if isinstance(model_for_save, PeftModel):
                print("Merging LoRA adapters into base weights …")
                model_for_save = model_for_save.merge_and_unload()
            # If not a PEFT model (full FT), it's already dense
            model_for_save.save_pretrained(merged_dir, safe_serialization=True)
            tok.save_pretrained(merged_dir)
            print("Merged weights saved to:", merged_dir)
        except Exception as e:
            print(f"[Warn] Failed to produce merged checkpoint: {e}")


if __name__ == "__main__":
    main()
