# my_SFT/SFT_pipeline.py

"""
Minimal-but-realistic Supervised Fine-Tuning (SFT) pipeline using PyTorch + HuggingFace 
(transformers + datasets), with:
  - JSONL dataset (instruction, optional input, output)
  - Prompt formatting for instruction-tuning
  - Label masking (train only on the assistant's response)
  - Custom PyTorch training loop (no HF Trainer)
  - Mixed precision, gradient accumulation, checkpointing
  - Perplexity eval and resume-from-checkpoint

Example dataset (train.jsonl):
{"instruction": "Sort numbers ascending.", "input": "9, 3, 4", "output": "3, 4, 9"}
{"instruction": "Translate to French.", "input": "Good night.", "output": "Bonne nuit."}
{"instruction": "What is 2+2?", "output": "4"}

Run:
python sft_from_scratch.py \
  --train_path train.jsonl --eval_path eval.jsonl \
  --model_name_or_path gpt2 \
  --output_dir ./sft-gpt2 \
  --max_length 512 --per_device_batch_size 4 \
  --gradient_accumulation_steps 8 --lr 2e-5 --epochs 3 \
  --warmup_ratio 0.03 --weight_decay 0.1 --log_every 20 \
  --bf16 --save_every 500

Multi-GPU:
torchrun --standalone --nproc_per_node=4 sft_from_scratch.py ...

(Note) If our tokenizer lacks a PAD token, we set PAD=eos.
"""

import argparse
import json
import math
import os
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from datasets import load_dataset

# ----- Prompt/template utilities -----
SYSTEM = (
    "You are a helpful, honest, and concise assistant.\n"
)

def format_example(example: Dict[str, str]) -> Dict[str, str]:
    instr = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    out = example.get("output", "").strip()

    if inp:
        prompt = (
            f"### System:\n{SYSTEM}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### System:\n{SYSTEM}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Response:\n"
        )

    target = out.strip()  # EOS will be appended by the collator
    return {"prompt": prompt, "target": target}


# ----- Dataset -----
class JSONLSFTDataset(Dataset):
    def __init__(self, path: str):
        self.rows: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    max_length: int
    label_mask_response_only: bool = True

    def __call__(self, features: List[Dict[str, str]]):
        input_texts: List[str] = []
        split_idxs: List[int] = []

        for ex in features:
            ft = format_example(ex)
            full = ft["prompt"] + ft["target"] + (self.tokenizer.eos_token or "")

            prompt_ids = self.tokenizer(
                ft["prompt"], 
                add_special_tokens=False, 
                truncation=True, 
                max_length=self.max_length,
            )["input_ids"]
            split_idxs.append(len(prompt_ids))
            input_texts.append(full)

        tokenized = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        if self.label_mask_response_only:
            for i in range(labels.size(0)):
                split = min(split_idxs[i], labels.size(1))
                labels[i, :split] = -100

        batch = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }
        return batch
    

# ----- Evaluation -----
@torch.no_grad()
def evaluate(model, dataloader, device, use_bf16=False, use_fp16=False):
    model.eval()

    # [CHANGE] Use token-weighted accumulators instead of averaging batch means.
    tw_numer = 0.0   # sum of (batch_mean_loss * valid_tokens_in_batch)
    tw_denom = 0     # sum of valid tokens across all batches

    # choose autocast dtype
    if use_bf16:
        autocast_dtype = torch.bfloat16
    elif use_fp16:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    for batch in dataloader:
        # move to device
        for k in batch:
            batch[k] = batch[k].to(device)

        # forward (no grad)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(**batch)
                loss = out.loss  # mean CE over *non-ignored* tokens in this batch
        else:
            out = model(**batch)
            loss = out.loss

        valid_tokens = int((batch["labels"] != -100).sum().item())

        if valid_tokens > 0:
            tw_numer += float(loss.detach()) * valid_tokens
            tw_denom += valid_tokens

    mean_loss = tw_numer / max(1, tw_denom)

    # perplexity from token-weighted loss
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    model.train()  # restore train mode
    return {"loss": mean_loss, "perplexity": ppl}


# ----- Training -----
def main():
    parser = argparse.ArgumentParser()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./sft-out")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--label_mask_response_only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print("Args:", args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Keep model config in sync with tokenizer
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.loss_type = "ForCausalLMLoss"
    model.config.pad_token_id = tokenizer.pad_token_id
    # Disable KV caching during training (faster + required when using GC)
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, backend="aot_eager")

    train_ds = JSONLSFTDataset(args.train_path)
    eval_ds = JSONLSFTDataset(args.eval_path) if args.eval_path else None

    collator = DataCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_mask_response_only=args.label_mask_response_only,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=max(1, args.per_device_batch_size // 2),
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    warmup_steps = int(args.warmup_ratio * max_train_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    global_step = 0
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        if rank == 0:
            print(f"Resumed from {args.resume_from} at step {global_step}, epoch {start_epoch}")

    scaler = torch.amp.GradScaler(device="cuda", enabled=args.fp16)
    use_bf16 = args.bf16 and torch.cuda.is_available()
    use_fp16 = args.fp16 and torch.cuda.is_available()

    model.train()


    for epoch in range(start_epoch, args.epochs):

        tw_numer = 0.0   # sum of (mini-batch loss * valid_tokens)
        tw_denom = 0     # sum of valid tokens
        accum_counter = 0 # counts only micro-batches that actually did backward()

        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            
            # Skip if nothing to learn (all labels == -100)
            valid_tokens = int((batch["labels"] != -100).sum().item())
            if valid_tokens == 0:
                continue

            if use_bf16:
                autocast_dtype = torch.bfloat16
            elif use_fp16:
                autocast_dtype = torch.float16
            else:
                autocast_dtype = None

            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    outputs = model(**batch)
                    loss = outputs.loss / args.gradient_accumulation_steps
                if use_fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()
            
            # Count only successful backward()s toward accumulation
            accum_counter += 1

            # outputs.loss is mean CE over valid tokens in this mini-batch
            mb_loss = float(outputs.loss.detach())
            tw_numer += mb_loss * valid_tokens
            tw_denom += valid_tokens

            if accum_counter % args.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if rank == 0 and global_step % args.log_every == 0:
                    avg_accum_loss = tw_numer / max(1, tw_denom)  # token-weighted mean loss
                    lr = scheduler.get_last_lr()[0]
                    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    print(f"epoch {epoch+1} step {global_step} | loss {avg_accum_loss:.4f} "
                            f"| lr {lr:.6e} | peak_mem {peak_mem:.1f} MB")
                    torch.cuda.reset_peak_memory_stats(device)

                # reset for the next accumulation window
                tw_numer = 0.0
                tw_denom = 0

                if rank == 0 and args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt_path = os.path.join(args.output_dir, f"ckpt_step{global_step}.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "tokenizer": tokenizer.__dict__.get("init_kwargs", {}),
                    }, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")

        # End epoch: evaluate & save
        if rank == 0 and eval_loader is not None:
            eval_metrics = evaluate(model, eval_loader, device, use_bf16=use_bf16, use_fp16=use_fp16)
            print(f"[Eval] epoch {epoch+1} | loss {eval_metrics['loss']:.4f} | ppl {eval_metrics['perplexity']:.2f}")

        if rank == 0:
            save_dir = os.path.join(args.output_dir, f"epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"Saved weights to {save_dir}")


    if rank == 0:
        print("Training complete.")


if __name__ == "__main__":
    main()
