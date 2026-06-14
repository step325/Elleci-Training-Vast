"""
Mamba-2 Checkpointed - VRAM Optimization
========================================

Wraps the Matmul-SSD kernel with Gradient Checkpointing (Activation Recomputation).
This trades compute (20-30% overhead) for massive VRAM savings (~90% of activation memory).

Usage:
    Enable via `config.mamba.use_checkpointing = True`
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .mamba2_optimized import Mamba2BlockMatmul, DifferentialMamba2BlockMatmul

class Mamba2BlockMatmulCheckpoint(Mamba2BlockMatmul):
    """
    Same as Mamba2BlockMatmul, but the memory-heavy 'chunk_forward' 
    is checkpointed (recomputed during backward).
    """
    
    def chunk_forward(self, x, dt, decay, B, C, state):
        """
        Wraps the super().chunk_forward in a checkpoint.
        
        Note on Checkpointing:
        - Inputs must require_grad for checkpointing to work (usually x does).
        - We pass use_reentrant=False for modern Pytorch stability.
        """
        
        # Checkpointing requires that at least one output has requires_grad
        # This is virtually guaranteed in a training loop.
        
        def run_fn(x, dt, decay, B, C, state):
             # Call the original implementation
             return super(Mamba2BlockMatmulCheckpoint, self).chunk_forward(x, dt, decay, B, C, state)

        # We need to wrap arguments in a way compatible with checkpoint
        # 'decay' is unused in the matmul kernel (it regenerates it), passing None or tensor works.
        
        # IMPORTANT: state usually does NOT require grad (it's updated in place or replaced)
        # Checkpointing works best with tensors that are part of the graph.
        
        return checkpoint(
            run_fn, 
            x, dt, decay, B, C, state,
            use_reentrant=False
        )

class DifferentialMamba2BlockCheckpoint(DifferentialMamba2BlockMatmul):
    """
    Differential Mamba-2 with Checkpointing logic.
    Inherits logical structure from DifferentialMamba2BlockMatmul,
    but we mixin the Checkpointing behavior by overriding chunk_forward 
    via the MRO (Method Resolution Order) or explicit composition.
    
    Since Python MRO can be tricky with parallel hierarchies, 
    we explicitly redefine chunk_forward here to use the checkpointing strategy.
    """
    
    def chunk_forward(self, x, dt, decay, B, C, state):
        # Same logic as Mamba2BlockMatmulCheckpoint
        
        def run_fn(x, dt, decay, B, C, state):
             # Call the original Matmul implementation (which is the parent of this class)
             # super(DifferentialMamba2BlockCheckpoint, self) -> DifferentialMamba2BlockMatmul
             # DifferentialMamba2BlockMatmul DOES NOT implement chunk_forward, it inherits from Mamba2BlockMatmul
             return super(DifferentialMamba2BlockCheckpoint, self).chunk_forward(x, dt, decay, B, C, state)

        return checkpoint(
            run_fn, 
            x, dt, decay, B, C, state, 
            use_reentrant=False
        )
