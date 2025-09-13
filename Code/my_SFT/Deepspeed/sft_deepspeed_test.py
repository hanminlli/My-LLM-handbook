# my_SFT/Deepspeed/sft_deepspeed.py
"""
DeepSpeed SFT example on GPT-NeoX-20B + UltraChat-200k.
Environment:
pip install \
  transformers==4.44.2 \
  trl==0.9.6 \
  datasets==2.19.0 \
  accelerate==0.31.0 \
  deepspeed==0.14.2 \
  bitsandbytes==0.43.3 \
  evaluate==0.4.1
"""

import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTTrainer, SFTConfig, setup_chat_format


# Settings 
MODEL_NAME = "EleutherAI/gpt-neox-20b"
DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
OUTPUT_DIR = "out_neox20b_ultra_ds3"
DEEPSPEED_CONFIG = "ds_zero3_offload.json"
'''
{
  "train_micro_batch_size_per_gpu": 1, # Each GPU processes 1 sample per forward/backward pass
  "gradient_accumulation_steps": 16, # Gradients are accumulated for 16 steps
  "bf16": { "enabled": true }, 
  "zero_optimization": {
    "stage": 3, # ZeRO3 partitions optimizer states, gradients and parameters accross GPUs
    "contiguous_gradients": true, # Keep gradient memory contiguous, avoids fragmentation
    "overlap_comm": true, # overlap communication (all-reduce) with computation.
    "reduce_scatter": true, # use reduce-scatter instead of all-reduce for efficiency
    "stage3_max_live_parameters": 1000000000, # internal heuristics for memory management: unlimited
    "stage3_max_reuse_distance": 1000000000, # unlimited
    "stage3_gather_16bit_weights_on_model_save": true, 
    # when saving checkpoints, gather model weights into FP16 (instead of sharded pieces).
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }, # parameters not in active use are offloaded to CPU memory (with pinned memory for speed)
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    } # optimizer states (Adam moments, etc.) stored on CPU, not GPU.
  },
  "gradient_clipping": 1.0, # Clips gradient norm at 1.0 to avoid exploding gradients.
  "wall_clock_breakdown": true
}
'''


SEED = 1337

MAX_STEPS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
LR = 1e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

SAVE_STEPS = 10
LOGGING_STEPS = 2
EVAL_STEPS = 10
EVAL_RATIO = 0.01

BF16 = True
GRADIENT_CHECKPOINTING = True
PACKING = True
MAX_SEQ_LENGTH = 2048

WANDB_PROJECT = "ds-sft"
WANDB_RUN_NAME = "neox20b-ultrachat"


def main():
    set_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    # only affects torch.matmul and operations built on it
    # torch.backends.cudnn.allow_tf32: affects cuDNN kernels 
    # mainly convolutions (nn.Conv2d, CNNs, some attention kernels using cuDNN).

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # rust version
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        device_map=None,  # DeepSpeed will handle placement
    )

    # TRL, adjust tokenizer and model to handle chat-style finetuning
    # Tokenizer side:
    #   Make sure the tokenizer has special tokens for chat interactions 
    #   (<|user|>, <|assistant|>, <|system|>, EOS token, etc., depending on the model).
    #   If those tokens don’t exist in the tokenizer vocabulary, add them.
    #   Updates the padding/truncation side to match chat datasets.
    # Model side:
    #   Resizes the model's embedding matrix so it matches the updated tokenizer vocabulary size.
    #   Ensures that the model recognizes and can generate using the added special tokens.
    model, tok = setup_chat_format(model, tok)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    full = load_dataset(DATASET_NAME, split=DATASET_SPLIT).select(range(2000))
    eval_size = max(100, int(len(full) * EVAL_RATIO))
    ds_train = full.select(range(len(full) - eval_size))
    ds_eval = full.select(range(len(full) - eval_size, len(full)))

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=SFTConfig(
            deepspeed=DEEPSPEED_CONFIG,
            bf16=BF16,
            learning_rate=LR,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=1,
            logging_steps=LOGGING_STEPS,
            save_steps=SAVE_STEPS,
            evaluation_strategy="steps",
            eval_steps=EVAL_STEPS,
            max_steps=MAX_STEPS,
            max_seq_length=MAX_SEQ_LENGTH,
            packing=PACKING,
            output_dir=OUTPUT_DIR,
            report_to=["wandb"],
        )
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    wandb.finish()

if __name__ == "__main__":
    main()
    