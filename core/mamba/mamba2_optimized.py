"""
Mamba-2 Optimized - Matmul-based SSD Implementation

This module provides a drop-in replacement for Mamba2Block that avoids 
the sequential 'scan' operation (and the external 'accelerated_scan' kernel),
instead using matrix multiplication (Matmul) to compute the State Space Duality.

Advantages:
1. Tensor Core Friendly: Uses torch.matmul / einsum which run on Tensor Cores (BF16/FP16).
2. Scan-Free: Removes the O(N) sequential dependency within chunks.
3. Compatibility: Pure PyTorch, works perfectly with INT2 training loop and Autograd.

Based on: "Transformers are SSMs: Generalized Models and Efficient Algorithms" (DAO et al.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

# Import base class to inherit structure
from .mamba2 import Mamba2Block

class Mamba2BlockMatmul(Mamba2Block):
    """
    Optimized Mamba-2 block using Matmul SSD (1-head Attention) instead of Scan.
    
    Instead of computing the state sequentially:
    h_t = decay * h_{t-1} + x_t
    
    We formulate the chunk computation as a masked matrix multiplication:
    y = Mask * (x * B) * C
    
    Where Mask is the decay matrix (Triangle). This is O(T^2) per chunk, 
    but since chunk_size (T) is small (e.g. 64), it is extremely efficient 
    and fully parallelizable on GPUs.
    """
    
    def chunk_forward(self, x, dt, decay, B, C, state):
        """
        Parallel SSD implementation using Matmul (Masked Attention).

        Args:
            x: [batch, chunk_size, n_heads, head_dim]
            dt: [batch, chunk_size, n_heads, head_dim]
            decay: [batch, chunk_size, n_heads, head_dim] - UNUSED (recomputed)
            B: [batch, chunk_size, d_state]
            C: [batch, chunk_size, d_state]
            state: [batch, n_heads, head_dim, d_state] - initial state from prev chunk

        Returns:
            y: [batch, chunk_size, n_heads, head_dim]
            new_state: [batch, n_heads, head_dim, d_state]
        """
        batch, cs, n_heads, head_dim = x.shape
        d_state = B.shape[-1]
        
        # 1. Compute cumulative decay (Log-space for stability)
        # A is scalar per head: [n_heads]
        A = -torch.exp(self.A_log.float())
        
        # dt: [B, cs, H, D]
        # dt_cumsum: [B, cs, H, D]
        dt_cumsum = torch.cumsum(dt, dim=1)
        
        # 2. Construct the decay Mask (L matrix) for the chunk
        # L[i, j] = exp(A * (dt_cumsum[i] - dt_cumsum[j])) for i >= j
        
        # Prepare terms for broadcasting
        # S: [B, cs, H, D]
        S = A.view(1, 1, n_heads, 1) * dt_cumsum
        
        # Permute S to [B, H, D, cs] for easier broadcasting
        S_perm = S.permute(0, 2, 3, 1) # [B, H, D, cs]
        
        S_i = S_perm.unsqueeze(-1) # [B, H, D, cs, 1]
        S_j = S_perm.unsqueeze(-2) # [B, H, D, 1, cs]
        
        M_log = S_i - S_j # [B, H, D, cs, cs]
        
        # Stability fix: Ensure M_log is not positive (which would mean growing instead of decaying)
        # S_i - S_j should be <= 0 for i >= j assuming correct A and dt.
        # Clamping prevents random init explosions.
        M_log = torch.clamp(M_log, max=0.0)
        
        # Apply causal mask (i >= j)
        indices = torch.arange(cs, device=x.device)
        # Fix: Lower Triangular Mask (i >= j)
        # rows (i) >= cols (j)
        causal_mask = indices.unsqueeze(1) >= indices.unsqueeze(0) # [cs, cs]
        
        M_exp = torch.exp(M_log)
        # Zero out non-causal
        M = M_exp * causal_mask.view(1, 1, 1, cs, cs) 
        
        # Cast M to input dtype (e.g. bf16) to save memory (cuts VRAM usage in half for stored activations)
        M = M.to(x.dtype)
        
        # M is now [B, H, D, cs, cs] (bf16/fp16)
        
        # 4. Compute State contribution from current chunk inputs
        # K = x * dt * B
        
        # Let U = x * dt  [B, cs, H, D]
        U = x * dt
        
        # Input term: I = U * B  (outer product at each step s)
        # I[b, s, h, d, p] = U[b, s, h, d] * B[b, s, p]
        
        # Matmul Logic:
        # 1. Prepare Term: [B, H, D, cs, P]
        #    Expand U: [B, H, D, cs, 1]
        #    Expand B: [B, 1, 1, cs, P]
        #    Term = U * B
        
        U_perm = U.permute(0, 2, 3, 1).unsqueeze(-1) # [B, H, D, cs, 1]
        B_perm = B.view(batch, 1, 1, cs, d_state)    # [B, 1, 1, cs, P]
        Term = U_perm * B_perm                       # [B, H, D, cs, P]
        
        # Mask: M [B, H, D, cs, cs]
        # THE MATMUL (Tensor Core Heavy Lifting)
        # [B,H,D,cs,cs] @ [B,H,D,cs,P] -> [B,H,D,cs,P]
        State_contrib = torch.matmul(M, Term)     # [B, H, D, cs, P]
        
        # 5. Add contribution from Initial State (Previous Chunk)
        # state_prev: [B, H, D, P]
        # decay_from_start: exp(A * dt_cumsum[t]) -> Decay for state_prev at time t
        # State_total[t] = State_contrib[t] + state_prev * decay_from_start[t]
        
        # decay_total: [B, cs, H, D] -> [B, H, D, cs, 1]
        # Note: S is [B, cs, H, D].
        decay_total = torch.exp(S).permute(0, 2, 3, 1).unsqueeze(-1) 
        
        State_total = State_contrib + state.unsqueeze(3) * decay_total
        
        # 6. Compute Output y
        # y[t] = sum_p State_total[t, p] * C[t, p]
        # y = sum(State * C, dim=-1)
        
        # C: [B, cs, P] -> [B, 1, 1, cs, P]
        C_perm = C.view(batch, 1, 1, cs, d_state)
        
        y_perm = (State_total * C_perm).sum(dim=-1) # [B, H, D, cs]
        
        y = y_perm.permute(0, 3, 1, 2) # [B, cs, H, D]
        
        # 7. Compute New State (for next chunk)
        # new_state is just the last timestep of State_total
        new_state = State_total[:, :, :, -1, :] # [B, H, D, P]
        
        return y, new_state

class DifferentialMamba2BlockMatmul(Mamba2BlockMatmul):
    """
    Differential Mamba-2 Block using Matmul implementation.
    See src.modules.mamba2.DifferentialMamba2Block for logic.
    """
    def __init__(self, config):
        Mamba2Block.__init__(self, config) # Initialize standard params
        
        # Copy-paste differential logic initialization
        self.diff_gate = nn.Linear(self.d_model, self.n_heads, bias=True)
        nn.init.zeros_(self.diff_gate.bias)
        nn.init.xavier_uniform_(self.diff_gate.weight, gain=0.1)
        self.gate_temperature = nn.Parameter(torch.ones(1))

    # Re-implement forward to use correct self.ssd_forward_cached call
    # which will then call our optimized chunk_forward
    def forward(self, x, use_cache=False, past_state=None):
        batch, seqlen, dim = x.shape

        # 1. Compute differential signal
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)
        diff = x - x_prev
        
        # 2. Gate
        gate_logits = self.diff_gate(diff)
        diff_gate = torch.sigmoid(gate_logits / self.gate_temperature)
        
        # 3. Standard Processing
        x_expanded = self.in_proj(x)
        x_expanded, z = x_expanded.chunk(2, dim=-1)
        
        x_conv = rearrange(x_expanded, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.act(x_conv)
        
        # 4. Modulate
        gate_expanded = repeat(diff_gate, 'b l h -> b l (h d)', d=self.head_dim)
        x_diff = x_conv * (1 + gate_expanded * 0.5)
        
        # 5. SSD (Optimized Matmul)
        if use_cache:
            y, present_state = self.ssd_forward_cached(x_diff, past_state)
        else:
            # We must ensure ssd_forward calls our chunk_forward
            # Mamba2Block.ssd_forward does call self.chunked_ssd, 
            # which calls self.chunk_forward. So inheriting structure works!
            y = self.ssd_forward(x_diff)
            present_state = None
            
        y = y * self.act(z)
        y = self.norm(y)
        output = self.out_proj(y)
        
        if use_cache:
            return output, present_state
        return output
