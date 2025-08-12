# my_distributed_MoE/moe_top2_multi_expert.py

import os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("USE_LIBUV", "0")
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class ExpertFFN(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x): 
        return self.net(x)


# ============================================================
# counts exchange + varlen all-to-all
# VARS:
#   send_counts : (world,) rows to send to each peer
#   recv_counts : (world,) rows we will receive from each peer
#   d           : feature dim (elements per row)
#   send_splits : per-peer ELEMENT counts to send
#   recv_splits : per-peer ELEMENT counts to receive
#   buf2d       : (rows, d) payload; we flatten to 1-D
# ============================================================
def exchange_counts(send_counts: torch.Tensor, group) -> torch.Tensor:
    """All-to-all exchange of ROW counts. Works on CUDA (NCCL) or CPU (gloo)."""
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    # Basically, in this all_to_all_single function, 
    # We are sending the send_counts[i] to the i-th rank from current rank r, which will land at recv_counts[r]
    # In this sense, recv_counts denotes the number of tokens we receive from rank r
    return recv_counts


def all_to_all_varlen(buf2d: torch.Tensor, send_counts: torch.Tensor, 
                      group, recv_counts: torch.Tensor | None = None):
    """
    Variable-size all-to-all for a 2-D buffer (rows, d).
    Returns (recv_buf2d, recv_counts). If recv_counts is given, reuses it.
    """
    device = buf2d.device
    d = buf2d.shape[-1] # The per ELEMENT dimension
    if recv_counts is None:
        recv_counts = exchange_counts(send_counts, group) # counts exchange if not provided
    
    # build splitting
    send_splits = (send_counts * d).tolist()
    recv_splits = (recv_counts * d).tolist()

    # flatten the payload for sending
    flat_send = buf2d.reshape(-1).contiguous() # the paylod for sending
    flat_recv = torch.empty(sum(recv_splits), device=device, dtype=buf2d.dtype) # the buffer for receiving

    # all to all send
    dist.all_to_all_single(flat_recv, flat_send, recv_splits, send_splits, group=group)
    # recv_split tells how much to receive from each rank, and send splits tells how much to send to each rank

    # reshape back to 2-D (rows, d)
    recv_rows = int(sum(recv_splits) // d)
    recv_buf2d = flat_recv.view(recv_rows, d)
    
    return recv_buf2d, recv_counts


# ============================================================
# For dispatch we must group rows by the rank that owns
# the chosen expert. We reuse the SAME indices to pack
# multiple aligned tensors (x, weight, origin metadata).
# VARS:
#   dest_rank : (N*,) int, which rank each row must go to
#   send_counts: rows per destination rank (world,)
#   offsets   : prefix-sum to place each bucket contiguously
#   idx       : indices of rows destined to a given rank
# ============================================================
def pack_by_dest_shared_idxs(*tensors, dest_rank: torch.Tensor, world_size: int):
    """
    Pack multiple aligned 2-D/1-D tensors by the same destination map.
    Each tensor's first dim must be equal (same rows).
    Returns: [packed_tensors...], send_counts, origin_row_indices
      - origin_row_indices lets us map packed rows back to original row ids if needed.
    """
    device = dest_rank.device
    rows = dest_rank.numel() # M, the total number of tensors to send
    send_counts = torch.bincount(dest_rank, minlength=world_size) # how much to send to each rank

    offsets = torch.zeros(world_size, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(send_counts[:-1], dim=0)

    # Prepare outputs 
    packed = []
    for t in tensors: 
        # tensors[0]: token embeddings (num_tokens, d_model), 
        # tensors[1]: token routing weights (num_tokens, 1),
        # tensors[2]: token metadata (num_tokens,)
        shape = list(t.shape)
        shape[0] = rows
        packed.append(torch.empty(shape, device=t.device, dtype=t.dtype))
    origin_row_indices = torch.empty(rows, dtype=torch.long, device=device)

    for r in range(world_size):
        idx = torch.nonzero(dest_rank == r, as_tuple=False).flatten() # the indices for sending to rank r
        k = idx.numel() # total number of sets of tensors
        if k == 0:
            continue
        s = offsets[r].item()
        # e = s + k

        # pack all tensors using the same idx
        for i, t in enumerate(tensors):
            packed[i].narrow(0, s, k).copy_(t.index_select(0, idx)) 
            # more efficient than packed[i][s:e] = t.index_select(0, idx)
        origin_row_indices.narrow(0, s, k).copy_(idx) # origin_row_indices[s:e] = idx

    return packed, send_counts, origin_row_indices


# ============================================================
# Top-2 dispatch (duplicate + route)
# Build the duplicated rows and all metadata we need on
# the receiver: weight, origin (rank & index), and which
# local expert (0 or 1) to run.
# VARS:
#   x           : (N_local, d_model) local tokens on this rank
#   top2_exp_id : (N_local, 2) global expert ids [0..E_total-1]
#   top2_weight : (N_local, 2) gate weights per expert
#   EPR         : experts per rank = 2 (fixed as per request)
#   owner_rank  : (N_local, 2) integer rank owning each expert
#   local_eid   : (N_local, 2) 0 or 1 -> which local expert on dest
#   origin_rank : (2*N_local,) this rank id, per duplicate
#   origin_idx  : (2*N_local,) original row id in this rank's batch
# ============================================================
def top2_dispatch(x: torch.Tensor, top2_exp_id: torch.Tensor, top2_weight: torch.Tensor,
                  experts_per_rank: int, group):
    """
    Duplicate rows for top-2, pack by destination rank, and all-to-all
    send the following aligned payloads:
      - token vectors
      - gate weights
      - origin_rank, origin_index
      - target local expert id at destination (0 or 1)
    Returns receive-side aligned tensors: recv_x, recv_w, recv_origin_rank,
    recv_origin_index, recv_local_eid, plus recv_counts and origin order.
    """
    device = x.device
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    N_local, d_model = x.shape

    # compute destination rank and local expert id
    owner_rank = (top2_exp_id // experts_per_rank).to(torch.long) # (N_local, 2) : destination rank
    local_eid  = (top2_exp_id %  experts_per_rank).to(torch.long) # (N_local, 2) in {0,1} : local expert id

    # duplicate rows for top-2 (because each token needs to be sent to two places)
    x2   = torch.cat([x, x], dim=0)  # (2N_local, d_model)
    w2   = top2_weight.reshape(-1, 1) # (2N_local, 1)
    dest = owner_rank.reshape(-1) # (2N_local,)
    le2  = local_eid.reshape(-1, 1) # (2N_local, 1)

    # origin metadata for reverse routing
    origin_rank = torch.full((2 * N_local, 1), rank, device=device, dtype=torch.long) # (2N_local, 1)
    origin_idx  = torch.arange(N_local, device=device, dtype=torch.long).repeat_interleave(2).view(-1, 1) 
    # (2N_local,1) : [0, 1, ...] --> [0, 0, 1, 1, ...]

    # pack by destination rank using shared indices for all buffers
    (packed_x, packed_w, packed_orank, packed_oidx, packed_le), send_counts, _ = \
        pack_by_dest_shared_idxs(x2, w2, origin_rank, origin_idx, le2,
                                 dest_rank=dest, world_size=world_size)

    # all-to-all all payloads using the SAME send_counts; exchange counts once and reuse
    recv_x,  recv_counts = all_to_all_varlen(packed_x,   send_counts, group, recv_counts=None)
    recv_w,  _           = all_to_all_varlen(packed_w,   send_counts, group, recv_counts=recv_counts)
    recv_orank, _        = all_to_all_varlen(packed_orank, send_counts, group, recv_counts=recv_counts)
    recv_oidx, _         = all_to_all_varlen(packed_oidx,  send_counts, group, recv_counts=recv_counts)
    recv_le, _           = all_to_all_varlen(packed_le,    send_counts, group, recv_counts=recv_counts)

    # squeeze metadata back to 1-D where appropriate
    recv_origin_rank  = recv_orank.squeeze(1).to(torch.long)   # (M,) M is the number of received tokens on this rank
    recv_origin_index = recv_oidx.squeeze(1).to(torch.long)    # (M,)
    recv_local_eid    = recv_le.squeeze(1).to(torch.long)      # (M,)

    return (recv_x, recv_w, recv_origin_rank, recv_origin_index, recv_local_eid, recv_counts, send_counts)


# ============================================================
# Run local experts (bucket by local_eid)
# On each rank we have TWO experts (0 and 1). We need to
# select rows for each, run the FFN, then place outputs
# back into the original receive order so downstream
# aggregation aligns with metadata.
# VARS:
#   recv_local_eid : (M,) 0 or 1 per received row
#   y_local        : (M, d_model) outputs aligned to recv order
# ============================================================
def run_two_local_experts(recv_x: torch.Tensor, recv_local_eid: torch.Tensor, experts: nn.ModuleList) -> torch.Tensor:
    """
    experts: length-2 ModuleList [expert0, expert1]
    """
    assert len(experts) == 2, "Exactly 2 experts are expected per rank."
    M, d_model = recv_x.shape
    y_local = torch.empty_like(recv_x)

    for le in (0, 1):
        idx = torch.nonzero(recv_local_eid == le, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        y_local[idx] = experts[le](recv_x.index_select(0, idx))
    
    return y_local

# ============================================================
# Pre-aggregate by (origin_rank, origin_index)
# A single origin token may have multiple contributions
# coming from this rank (e.g., if both local experts got
# copies for the same token). We SUM them before the
# reverse all-to-all. We also bucket by destination rank
# (which equals origin_rank).
# VARS:
#   recv_origin_rank  : (M,) original rank id for each row
#   recv_origin_index : (M,) original index on that rank
#   key64             : (M,) composite key = (rank << 32) | index
#   uniq_key, inv     : unique keys + inverse map for grouping
#   agg_vals          : (K, d) summed contributions per key
#   dest_rank_back    : (K,) = origin_rank
#   dest_index_back   : (K,) = origin_index
#   send_counts_back  : (world,) rows we send to each peer
#   packed_vals/idx   : concatenated per-dest in rank order
# ============================================================
def preaggregate_and_pack_for_return(
        y_local             : torch.Tensor,
        recv_w              : torch.Tensor,
        recv_origin_rank    : torch.Tensor,
        recv_origin_index   : torch.Tensor,
        group
    ):
    """
    Weighted-sum by (origin_rank, origin_index), then pack by dest rank for reverse a2a.
    Returns packed_vals, packed_idx, send_counts_back.
    """
    device = y_local.device
    world_size = dist.get_world_size(group=group)

    # apply gates weights
    contrib = y_local * recv_w  # (M, d)

    # build composite (rank, index) 64-bit keys for grouping
    key64 = (recv_origin_rank.to(torch.int64) << 32) | recv_origin_index.to(torch.int64)
    uniq_key, inv = torch.unique(key64, return_inverse=True, sorted=False)
    # higher 32 bits for original rank and lower bits for original index
    # unique returns all distinct key64 pair, inv is an integer array mapping each original key64 to its position in uniq_key.

    # aggregate contributions per unique key
    K, d_model = uniq_key.numel(), y_local.shape[1]
    agg_vals = torch.zeros((K, d_model), device=device, dtype=y_local.dtype)
    agg_vals.index_add_(0, inv, contrib)

    # recover destination routing info
    dest_rank_back  = (uniq_key >> 32).to(torch.long) # higher 32
    dest_index_back = (uniq_key & ((1 << 32) - 1)).to(torch.long) # lower 32

    # bucket by dest rank for reverse all-to-all 
    send_counts_back = torch.bincount(dest_rank_back, minlength=world_size) 
    offsets = torch.zeros(world_size, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(send_counts_back[:-1], dim=0)

    # buffers for packing
    packed_vals = torch.empty_like(agg_vals)
    packed_idx  = torch.empty_like(dest_index_back)

    for r in range(world_size):
        idx = torch.nonzero(dest_rank_back == r, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        s = offsets[r].item()
        e = s + idx.numel()
        packed_vals[s:e] = agg_vals.index_select(0, idx)
        packed_idx[s:e]  = dest_index_back.index_select(0, idx)
    
    return packed_vals, packed_idx, send_counts_back


# ============================================================
# Reverse all-to-all and scatter-add
# Send packed contributions and origin indices back to
# the origin ranks. On the origin, place them into the
# output tensor at the correct positions, summing if the
# same token receives from multiple peers.
# VARS:
#   send_counts_back : (world,) rows per dest rank
#   org_back         : (rows_back,) origin indices on this origin rank
#   vals_back        : (rows_back, d) values aligned with org_back
#   y_out            : (N_local, d) final outputs in original order
# ============================================================
def return_and_scatter(
        packed_vals     : torch.Tensor,
        packed_idx      : torch.Tensor,
        send_counts_back: torch.Tensor,
        group,
        N_local         : int,
        d_model         : int
    ):
    """
    Reverse all to all: send (values, origin_index) back. Scatter-add into y_out.
    """
    # The index stream is 1-d, send it as (rows, 1)
    idx2d = packed_idx.view(-1, 1)  # stays torch.long

    # Single counts exchange reused for both streams
    recv_counts_back = exchange_counts(send_counts_back, group)

    vals_back, _    = all_to_all_varlen(packed_vals, send_counts_back, group, recv_counts=recv_counts_back)
    org_back2d, _   = all_to_all_varlen(idx2d, send_counts_back, group, recv_counts=recv_counts_back)
    org_back        = org_back2d.squeeze(1).to(torch.long)

    # Place into output (sum if multiple peers send the same origin index)
    y_out = torch.zeros(N_local, d_model, device=packed_vals.device, dtype=packed_vals.dtype)
    y_out.index_add_(0, org_back, vals_back)
    return y_out


# ============================================================
# MoE layer wrapper (tie everything together)
#   1) dispatches,
#   2) runs the two local experts,
#   3) aggregates + returns,
#   4) gives you outputs aligned with the input order.
# ============================================================
class MoETop2TwoExpertsPerRank(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff), ExpertFFN(d_model, d_ff)]) # 2 experts per rank

    def forward(
        self, 
        x_local: torch.Tensor,
        top2_exp_id: torch.Tensor,
        top2_weight: torch.Tensor,
        group
    ) -> torch.Tensor:
        """
        x_local:      (N_local, d_model) tokens owned by this rank
        top2_exp_id:  (N_local, 2) global expert ids chosen by the gate
        top2_weight:  (N_local, 2) gate weights per chosen expert
        """
        d_model = x_local.shape[-1]
        EPR = 2  # experts per rank

        # Dispatch to expert-owning ranks
        (recv_x, recv_w, recv_orank, recv_oidx, recv_le, recv_counts, send_counts) = top2_dispatch(
            x_local, top2_exp_id, top2_weight, EPR, group
        )

        # Run our two local experts
        y_local = run_two_local_experts(recv_x, recv_le, self.experts)
        
        # Pre-aggregate by (origin_rank, origin_index) and pack for return
        packed_vals, packed_idx, send_counts_back = preaggregate_and_pack_for_return(
            y_local, recv_w, recv_orank, recv_oidx, group
        )

        # Reverse all-to-all and scatter-add to original order
        y_out = return_and_scatter(
            packed_vals, packed_idx, send_counts_back, group,
            N_local=x_local.shape[0], d_model=d_model
        )
        return y_out


# ============================================================
# Test:
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 moe_top2_multi_expert.py
# ============================================================
def test_MoE_top2_a2a():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank  = int(os.environ.get("RANK", "0"))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    backend = "nccl" if (device.type == "cuda" and world > 1) else "gloo" # gloo for CPU
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
    if device.type == "cuda":
        torch.cuda.set_device(rank) # For this process (rank), use GPU with ID = rank as the default CUDA device

    torch.manual_seed(0 + rank) # different ranks get different seeds.

    # Hyperparams
    d_model, d_ff = 20, 80
    EPR = 2  # experts per rank (fixed)
    total_experts = world * EPR

    # Local batch (flatten B*L outside this example)
    N_local = 8
    x = torch.randn(N_local, d_model, device=device)

    # Fake gate: choose two distinct experts + weights per token
    logits      = torch.randn(N_local, total_experts, device=device)
    top2        = torch.topk(logits, k=2, dim=-1)          # values, indices
    top2_exp_id = top2.indices                             # (N_local, 2)
    gate        = torch.softmax(top2.values, dim=-1)       # (N_local, 2)

    # MoE layer
    moe = MoETop2TwoExpertsPerRank(d_model, d_ff).to(device)

    y = moe(x, top2_exp_id, gate, dist.group.WORLD)
    with torch.no_grad():
        # here we at least check finite and shape
        assert torch.isfinite(y).all()

    assert y.shape == x.shape
    if rank == 0:
        print("TEST OK: y shape", y.shape)

    dist.barrier()
    dist.destroy_process_group()

# ============================================================
# Test on windows
# ============================================================
def _worker(rank, world_size, port):
    import os
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # pure c10d init via env://
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, init_method="env://")

    # force CPU on Windows
    torch.manual_seed(1234 + rank)

    test_MoE_top2_a2a()  

def run_spawn(world_size=2, port=29500):
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)

if __name__ == "__main__":
    run_spawn(world_size=2, port=29500)
    # using command python moe_top2_alltoall.py