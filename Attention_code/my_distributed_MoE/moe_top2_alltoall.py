# my_distributed_MoE/moe_top2_alltoall.py

import torch
import torch.nn as nn
import torch.distributed as dist

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


def dup_dispatch_top2(x, top2_ranks, top2_weights, group):
    """
    x: (N, d)
    top2_ranks: (N, 2) destination ranks for each token
    top2_weights: (N, 2) combine weights per destination
    group: the distributed process group
    Returns:
      recv_x: tokens to process on this rank
      recv_w: weights aligned with recv_x
      recv_origin: origin indices aligned with recv_x
      recv_counts, send_counts for the return step
    """
    world = dist.get_world_size(group=group) # num_nodes × gpus_per_node = total number of processes participate in the training
    device = x.device
    N, d = x.shape

    # Duplicate rows
    x2 = torch.cat([x, x], dim=0)  # (2N, d) two copies of every token
    w2 = top2_weights.reshape(-1, 1) # (2N, 1) matching weights
    dest = top2_ranks.reshape(-1) # (2N,) flatten destination ranks
    origin = torch.arange(N, device=device).repeat_interleave(2) # (2N,) : [0, 0, 1, 1, ...] original token IDs

    send_counts = torch.bincount(dest, minlength=world)  # send_counts[r] = how many vectors to be sent to rank r
    offsets = torch.zeros(world, dtype=torch.long, device=device) # all zeros
    offsets[1:] = torch.cumsum(send_counts[:-1], dim=0) # cumsum from the second pos

    # Allocate buffers for token data, weights, and origin indices in the packed order.
    packed_x   = torch.empty((2*N, d), device=device, dtype=x.dtype) 
    packed_w   = torch.empty((2*N, 1), device=device, dtype=x.dtype)
    packed_org = torch.empty(2*N, dtype=torch.long, device=device)
    
    for r in range(world):
        idx = torch.nonzero(dest == r, as_tuple=False).flatten() 
        # returns an 2D tensor [num_matchs, 1] which contain the indices where dest == r, and then flatten
        k = idx.numel() 
        if k == 0: continue
        s, e = offsets[r].item(), offsets[r].item() + k
        packed_x[s:e]   = x2.index_select(0, idx) # select the tensors whose destination is rank == r
        packed_w[s:e]   = w2.index_select(0, idx)  # select the weights (gates)
        packed_org[s:e] = origin.index_select(0, idx) # select to store the origin

    # all-to-all on x and weights and origin indices (send counts shared)
    def a2a(buf2d):
        from_counts = send_counts
        # 1) exchange counts
        # We need to know how many rows we will receive from every sender
        recv_counts = torch.empty_like(from_counts) # same shape, dtype, and device
        dist.all_to_all_single(recv_counts, from_counts, group=group)
        # ---------------------------------------------------------------------------------------------------------
        # For a variable-sized all_to_all_single on the data (tokens), we must pass send splits and recv splits.
        # We already know our send splits (from_counts * d where d is feature dim).
        # But we don’t know our recv splits until every other rank tells us how many items they’re sending to us.
        
        # all_to_all_signle: no splits list are provided, so Pytorch uses equal split, with length == world_size
        # that's 1 scalar per peer. 

        # Effect: (on rank r)
        # We send from_counts[j] to rank j (one scalar per peer)
        # We receive one scalar from each peer i, which lands in recv_counts[i].
        # After the call, on rank r: recv_counts[i]: how many items rank i will send to rank r in the upcoming 
        # data all-to-all.
        # ---------------------------------------------------------------------------------------------------------
        # 2) flatten and compute element splits
        dflat = buf2d.reshape(-1) # Flattens the 2-D buffer to 1D-tensor to receive
        d = buf2d.shape[-1] # Flattens the 2-D buffer buf2d (shape (rows, d)) into a 1-D tensor.
        # rows: How many items we are about to send to all peers combined in this all-to-all call,
        # all_to_all_single expects counts in elements (flattened),
        send_splits = (from_counts * d).tolist() # turn to list to fit the API; elements to send to each peer.
        recv_splits = (recv_counts * d).tolist() # elements to receive from each peer.

        # 3) allocate output and do variable-size all_to_all
        out = torch.empty(sum(recv_splits), device=device, dtype=buf2d.dtype) 
        # 1-D buffer to hold all the data that will be received from every peer
        dist.all_to_all_single(out, dflat, recv_splits, send_splits, group=group)
        # out: the receive buffer (1-D tensor) — must be preallocated with enough space.
        # dflat: the send buffer (1-D tensor) — our local data to send.
        # recv_splits: list/tuple of how many elements to expect from each rank.
        # send_splits: list/tuple of how many elements to send to each rank.
        # group: the communication group (usually dist.group.WORLD).
        return out.view(-1, d), recv_counts
    
    recv_x, recv_counts = a2a(packed_x) # The embeddings this rank’s expert should process. 
    recv_w, _           = a2a(packed_w) # The per-token gate weights correspond to packed_x
    recv_org, _         = a2a(packed_org.unsqueeze(1)) 
    recv_org            = recv_org.squeeze(1).to(torch.long) # The origin of each token
    return recv_x, recv_w, recv_org, recv_counts, send_counts





if __name__ == "__main__":
    pass