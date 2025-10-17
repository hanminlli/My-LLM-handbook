# ./my_RLHF/proximal_policy_optimization.py


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
    max_prompts: int = 1000

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
        logits = self.lm.lm_head(last_hidden)  # (B, L, V)
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
    prompts = [p.strip() for p in prompts if isinstance(p, str) and len(p.strip()) > 0]
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
    texts = [f"User: {p}\nAssistant: {r}" for p, r in zip(prompts, responses)]
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


def compute_gae_with_bootstrap(rewards, values, mask, gamma, lam):
    '''
    GAE for variable-length responses with bootstrap value=0 after the last response token.
    rewards/values/mask: (B, T_full - 1) aligned to positions where policy/value/logprob are defined.
    mask is 1 on response tokens, 0 elsewhere.
    Positions before response (prompt) are zero-masked already.
    Returns advantages, returns with same shape as inputs (zeros outside response).
    '''
    B, T = rewards.shape # we have already spread out the rewards
    adv = torch.zeros_like(rewards) # A_t
    ret = torch.zeros_like(rewards) # G_t

    # For each sample, run a backward pass across response slice only
    for i in range(B):
        valid_idx = mask[i].nonzero(as_tuple=False).squeeze(-1)
        # nonzero returns (num_valid, 1)
        if valid_idx.numel() == 0:
            continue
        start, end = int(valid_idx[0].item()), int(valid_idx[-1].item())
        # Continguous
        r = rewards[i, start : end + 1]
        v = values[i, start : end + 1]
        Tr = r.size(0)
        adv_i = torch.zeros_like(r)
        lastgaelam = torch.zeros(1, device=r.device, dtype=r.dtype) # 0
        next_value = torch.zeros(1, device=r.device, dtype=r.dtype) # 0 
        for t in reversed(range(Tr)):
            delta = r[t] + gamma * next_value - v[t] # TD-redisual
            lastgaelam = delta + gamma * lam * lastgaelam # Advantage estimation
            adv_i[t] = lastgaelam
            next_value = v[t] # for the previous step
        ret_i = adv_i + v # add A and V to compute G
        adv[i, start : end + 1] = adv_i
        ret[i, start : end + 1] = ret_i

    return adv, ret
    

# Build models & tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.policy_id, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # safe default and align with generate()

dtype = torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() else None
policy = PolicyWithValueHead(args.policy_id, dtype=dtype, device_map=args.device_map) 
ref = clone_frozen_ref(policy) # pi_ref

optim = torch.optim.AdamW(
    [
        {"params": policy.lm.parameters(), "lr": args.learning_rate},
        {"params": policy.value_head.parameters(), "lr": args.learning_rate},
    ],
    betas=(0.9, 0.95),
    weight_decay=0.01
)

kl_coef = args.init_kl_coef


# Rollout: prompts -> generations -> rewards -> training batch
def rollout(batch_prompts: List[str]):
    # format chats
    chats = [build_chat(p) for p in batch_prompts] # List[Dict[str, str]]
    prompts_texts = [render_chat(tokenizer, m) for m in chats] # List[str]
    enc = tokenizer(
        prompts_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max        
    )
    prompt_ids   = enc["input_ids"].to(policy.lm.device) # batched prompts tokens
    prompt_mask = enc["attention_mask"].to(policy.lm.device) # padding mask
    prompt_lens  = [int(m.sum().item()) for m in prompt_mask] # true prompt lengths
    # recall that padding happens on the left

    # generate responses
    with torch.no_grad():
        gen = policy.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True, # not always picking the same token with the maximum probability
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=policy.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ) # (B, T_full), right padded (there is no left padding here)

    # build full inputs, labels, and response mask
    input_ids_list, attn_list = [], []
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    B_full, L_full = gen.shape
    for i in range(B_full):
        full = gen[i] # (L_full,)
        # PAD-aware attention mask (1 for real tokens, 0 for PAD)
        attn = (full != pad_id).long()
        input_ids_list.append(full)
        attn_list.append(attn)

    # stack and pad for safety
    input_ids = pad_to_same_length(input_ids_list, pad_id)    # (B, L_full)
    attention_mask = pad_to_same_length(attn_list, 0)         # (B, L_full)

    # Shifted labels
    labels_shifted = input_ids[:, 1:].clone()  # (B, L_full - 1)
    B, Tm1 = labels_shifted.shape
    # Response mask aligned with (B, L_full - 1): 1 from (p_len-1) .. up to last target token BEFORE EOS/PAD
    resp_mask_full = torch.zeros((B, Tm1), device=input_ids.device, dtype=torch.float32)
    for i, p_len in enumerate(prompt_lens):
        valid = (attention_mask[i] == 1).nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        last_real = valid[-1].item()
        # If EOS exists, exclude it: set last target token = EOS_index - 1
        eos_pos = (input_ids[i] == eos_id).nonzero(as_tuple=False)
        if eos_pos.numel() > 0:
            last_target = int(eos_pos[0].item()) - 1
        else:
            last_target = last_real

        start = max(p_len - 1, 0)
        end_exclusive = max(last_target, start) 
        if end_exclusive > start:
            resp_mask_full[i, start:end_exclusive] = 1.0

    # Compute old logprobs/values (actor) and ref logprobs (reference)
    with torch.no_grad():
        # in the previous generation pass there are no information stored
        logits, values = policy(input_ids, attention_mask)
        # logits: (B, L_full, V), values: (B, L_full), generated by the value head
        old_logprobs_full = policy.token_logprobs(logits[:, :-1], labels_shifted) # (B, L_full - 1)

        ref_out = ref.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        ref_logits = ref.lm_head(ref_out.last_hidden_state) # (B, L_full, V)
        ref_logp_full = F.log_softmax(ref_logits[:, :-1], dim=-1) # (B, L_full - 1)
        ref_selected = torch.gather(ref_logp_full, -1,  labels_shifted.unsqueeze(-1)).squeeze(-1) # (B, L_full - 1)
        values_full = values[:, :-1] # (B, L_full - 1)

    # Decode responses (for reward)
    responses = []
    for i in range(B):
        p_len = prompt_lens[i]
        resp_text = tokenizer.decode(input_ids[i, p_len:], skip_special_tokens=True).strip()
        responses.append(resp_text)
        # The RL policy and reward model often have different tokenizers


    # Scalar reward per sample
    rewards_scalar = reward_fn_rm(batch_prompts, responses).to(input_ids.device) # (B, )
    resp_lengths = resp_mask_full.sum(dim=1).clamp_min(1.0) # (B, )
    rewards_per_token = (rewards_scalar / resp_lengths).unsqueeze(-1) * resp_mask_full # (B, L_full - 1)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels_shifted": labels_shifted,
        "old_logprobs": old_logprobs_full,
        "ref_logprobs": ref_selected,
        "values": values_full,
        "resp_mask": resp_mask_full,
        "rewards": rewards_per_token,
        "prompts": batch_prompts,
        "responses": responses,
    }

    return batch


# PPO Update (with GAE bootstrap + gradient-carrying entropy)
def ppo_update(batch, kl_coef, target_kl=0.1):
    input_ids = batch["input_ids"]                  # (B, L_full)
    attention_mask = batch["attention_mask"]        # (B, L_full)
    labels_shifted = batch["labels_shifted"]        # (B, L_full-1)
    old_logprobs = batch["old_logprobs"].detach()   # (B, L_full-1)
    ref_logprobs = batch["ref_logprobs"].detach()   # (B, L_full-1)
    values_old = batch["values"].detach()           # (B, L_full-1)
    rewards = batch["rewards"]                      # (B, L_full-1)
    resp_mask = batch["resp_mask"]                  # (B, L_full-1)

    B, T = old_logprobs.shape # T = T_full - 1

    # Compute GAE/Returns on response tokens with bootstrap
    # Values are aligned with (B, T_full-1). We'll pass rewards/values/resp_mask as-is.
    adv, rets = compute_gae_with_bootstrap(rewards, values_old, resp_mask, args.gamma, args.lam)

    # Advantage normalization over response tokens only
    mean_adv = masked_mean(adv, resp_mask) 
    std_adv = torch.sqrt(masked_mean((adv - mean_adv) ** 2, resp_mask) + 1e-8)
    adv_norm = (adv - mean_adv) / (std_adv + 1e-8)
    # In practice, the scale of A_t can vary greatly, within batches, between batches 
    # we normalize to make it roughly zero-mean and unit variance within the batch

    # Build a flat index mask for response positions
    flat_mask = resp_mask.reshape(-1) > 0.0 # converted to boolean
    valid_idx = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    N = valid_idx.numel()
    if N == 0:
        return {
            "approx_kl": 0.0,
            "reward_mean": 0.0,
            "adv_mean": 0.0,
            "kl_coef": kl_coef,
        }
    
    # Minibatch sampler over response positions
    idx = torch.arange(N, device=input_ids.device)
    for epoch in range(args.ppo_epochs):
        perm = idx[torch.randperm(N)] # random reshuffling
        # randomize the training order of response tokens in epoch, prevent bias from sequence order
        for start in range(0, N, args.minibatch_size):
            sel = valid_idx[perm[start:start+args.minibatch_size]]

            # Recompute current policy output
            cur_logits, cur_value_full = policy(input_ids, attention_mask) # (B, L, V), (B, L)
            cur_logprobs_full = policy.token_logprobs(cur_logits[:, :-1], labels_shifted) # (B, L_full - 1)
            cur_values = cur_value_full[:, :-1]
            cur_logprobs = cur_logprobs_full.reshape(-1)[sel]
            old_lp = old_logprobs.reshape(-1)[sel]
            ref_lp = ref_logprobs.reshape(-1)[sel]
            values_cur = cur_values.reshape(-1)[sel]
            adv_mb = adv_norm.reshape(-1)[sel]
            returns_mb = rets.reshape(-1)[sel]
        
            # PPO clipped objective
            ratio = torch.exp(cur_logprobs - old_lp)
            unclipped = ratio * adv_mb
            clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1 + args.clip_range) * adv_mb
            ppo_loss = - torch.mean(torch.min(unclipped, clipped))

            # Value loss 
            v_loss = F.mse_loss(values_cur, returns_mb)

            # Approximate KL vs reference on chosen actions
            approx_kl = torch.mean(cur_logprobs - ref_lp)
            kl_loss = kl_coef * approx_kl

            # Entropy Bonus
            logp_full = F.log_softmax(cur_logits[:, :-1], dim=-1) # (B, L_full - 1, V)
            p_full = logp_full.exp()
            ent_full = - (p_full * logp_full).sum(-1)
            ent_mb = torch.mean(ent_full.reshape(-1)[sel])
            ent_bonus = -args.ent_coef * ent_mb

            loss = ppo_loss + args.vf_coef * v_loss + kl_loss + ent_bonus

            optim.zero_grad(set_to_none=True) # Frees memory and avoids unnecessary writes; slightly faster
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            # If the total gradient norm across all parameters exceeds 1.0, 
            # scale all gradients down proportionally so that their overall norm = 1.0.
            # In PPO (and RLHF in general), policy gradients can occasionally spike, 
            # gradient clipping can caps those updates, ensuring training stays stable and bounded
    
    # Stats on the full response region
    with torch.no_grad():
        logits_after, _ = policy(input_ids, attention_mask)
        cur_lp_full = policy.token_logprobs(logits_after[:, :-1], labels_shifted)
        approx_kl_full = (cur_lp_full - ref_logprobs)
        approx_kl_mean = masked_mean(approx_kl_full, resp_mask).item()
        reward_mean = masked_mean(rewards, resp_mask).item()
        adv_mean = masked_mean(adv_norm, resp_mask).item()
    
    # Simple KL controller to keep KL near target
    if approx_kl_mean > target_kl * 1.5:
        kl_coef *= 1.5
    elif approx_kl_mean < target_kl / 1.5:
        kl_coef /= 1.5
    kl_coef = float(max(1e-4, min(1.0, kl_coef)))
    return {
        "approx_kl": approx_kl_mean,
        "reward_mean": reward_mean,
        "adv_mean": adv_mean,
        "kl_coef": kl_coef,
    }


# Main
def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    global kl_coef
    for step in range(args.steps):
        prompts = sample_batch(args.batch_size)
        batch = rollout(prompts)
        stats = ppo_update(batch, kl_coef, target_kl=0.1)
        kl_coef = stats["kl_coef"]
        print(
            f"[step {step}] "
            f"reward={stats['reward_mean']:.4f} "
            f"KL≈{stats['approx_kl']:.4f} "
            f"adv_mean={stats['adv_mean']:.3f} "
            f"kl_coef={kl_coef:.4f}"
        )
    outdir = "ppo_scratch_out"
    os.makedirs(outdir, exist_ok=True)
    policy.lm.save_pretrained(outdir)
    torch.save(policy.value_head.state_dict(), os.path.join(outdir, "value_head.pt"))
    tokenizer.save_pretrained(outdir)
    print(f"Saved to: {outdir}")



if __name__ == "__main__":
    main()
