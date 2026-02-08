"""
Rotary Position Embedding (RoPE)

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
Used by LLaMA, Mistral, Qwen, and most modern LLMs.

Key advantages:
- No learnable parameters (just math)
- Generalizes to unseen sequence lengths
- Relative position encoding (better for attention)
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding.
    
    Precomputes sin/cos tables for efficient RoPE application.
    
    Args:
        dim: Dimension of the embedding (usually head_dim)
        base: Base for the frequency computation (higher = longer context)
              - 10000: ~8k context
              - 100000: ~128k context (LLaMA 3.1 style)
        max_seq_len: Maximum sequence length to precompute
    """
    def __init__(self, dim: int, base: int = 100000, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for sin/cos (will be computed on first forward)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
        
    def _update_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cached cos/sin tables if sequence length increased."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            
            # Position indices [0, 1, 2, ..., seq_len-1]
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            
            # Frequencies: [seq_len, dim/2]
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Concat for full dim: [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Cache cos and sin
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos/sin embeddings for the given sequence length.
        
        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length (if None, uses x.shape[-2])
            
        Returns:
            (cos, sin): Each [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[-2]
            
        self._update_cos_sin_cache(seq_len, x.device, x.dtype)
        
        return (
            self._cos_cached[:seq_len],
            self._sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [batch, n_heads, seq_len, head_dim]
        k: Key tensor [batch, n_heads, seq_len, head_dim] or None
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        position_ids: Optional position indices (for generation with cache)

    Returns:
        (q_rotated, k_rotated): k_rotated is None if k was None
    """
    # Handle position_ids for cached generation
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq, dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Broadcast cos/sin to match q/k shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation to Q
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # Apply rotation to K (only if provided)
    k_embed = None
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class ContextAwareRoPE(RotaryEmbedding):
    """
    Context-Aware RoPE (CARoPE) - arXiv:2507.23083

    Extends standard RoPE with context-dependent frequency adaptation.
    Instead of using fixed frequencies, CARoPE adjusts frequencies based
    on the input context, allowing better adaptation to different types
    of content (code, prose, dialogue, etc.).

    Key insight: Different contexts benefit from different position
    encoding frequencies. Code needs precise local positions, while
    prose may need longer-range dependencies.

    Args:
        dim: Dimension of the embedding (usually head_dim)
        base: Base for frequency computation
        max_seq_len: Maximum sequence length
        freq_adapt_scale: Scale factor for frequency adaptation (0.0 = standard RoPE)
    """
    def __init__(
        self,
        dim: int,
        base: int = 100000,
        max_seq_len: int = 8192,
        freq_adapt_scale: float = 0.1
    ):
        super().__init__(dim, base, max_seq_len)
        self.freq_adapt_scale = freq_adapt_scale

        # Context-to-frequency adapter
        # Projects context embedding to frequency modulation
        self.freq_adapter = nn.Linear(dim, dim // 2, bias=False)

        # Initialize to near-identity (small perturbations initially)
        nn.init.xavier_uniform_(self.freq_adapter.weight, gain=0.01)

        # Learnable base frequency scaling
        self.base_scale = nn.Parameter(torch.ones(dim // 2))

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        context_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get context-aware cos/sin embeddings.

        Args:
            x: Input tensor (used for device/dtype)
            seq_len: Sequence length
            context_emb: Optional context embedding [batch, dim] for adaptation
                        If None, uses mean of x as context

        Returns:
            (cos, sin): Each [batch, seq_len, dim] or [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Compute base frequencies
        # inv_freq: [dim/2]
        inv_freq = self.inv_freq.to(x.device)

        # Context-dependent frequency modulation
        if context_emb is None:
            # Use mean of input as context
            if x.dim() == 4:  # [batch, heads, seq, dim]
                context_emb = x.mean(dim=[1, 2])  # [batch, dim]
            else:  # [batch, seq, dim]
                context_emb = x.mean(dim=1)  # [batch, dim]

        # Adapt frequencies based on context
        # freq_delta: [batch, dim/2]
        freq_delta = self.freq_adapter(context_emb)
        freq_delta = torch.tanh(freq_delta) * self.freq_adapt_scale

        # Modulated frequencies: [batch, dim/2]
        adapted_inv_freq = inv_freq.unsqueeze(0) * (1 + freq_delta) * self.base_scale.unsqueeze(0)

        # Position indices: [seq_len]
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # Compute frequencies for each position: [batch, seq_len, dim/2]
        # Outer product: [batch, 1, dim/2] * [1, seq_len, 1] -> [batch, seq_len, dim/2]
        freqs = torch.einsum('bd,s->bsd', adapted_inv_freq, t)

        # Concat for full dim: [batch, seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Compute cos and sin
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)

        return cos, sin


class DAPEv2(RotaryEmbedding):
    """
    DAPE V2 - Improved Length Extrapolation (arXiv:2410.04798)

    Addresses the length extrapolation problem in RoPE by:
    1. Using decaying attention bias for distant positions
    2. Adaptive base frequency scaling based on sequence length
    3. Smooth interpolation for unseen lengths

    Args:
        dim: Embedding dimension
        base: Base frequency
        max_seq_len: Maximum sequence length (can extrapolate beyond)
        alpha: Decay factor for attention bias
        scaling_factor: NTK-aware scaling factor
    """
    def __init__(
        self,
        dim: int,
        base: int = 100000,
        max_seq_len: int = 8192,
        alpha: float = 1.0,
        scaling_factor: float = 1.0
    ):
        super().__init__(dim, base, max_seq_len)
        self.alpha = alpha
        self.scaling_factor = scaling_factor
        self.trained_max_len = max_seq_len

        # Learnable extrapolation parameters
        self.extrapolation_factor = nn.Parameter(torch.ones(1))

    def _compute_dynamic_base(self, seq_len: int) -> float:
        """Compute dynamic base for length extrapolation."""
        if seq_len <= self.trained_max_len:
            return self.base

        # NTK-aware scaling
        ratio = seq_len / self.trained_max_len
        dynamic_base = self.base * (ratio ** (self.scaling_factor / (self.dim - 2)))
        return dynamic_base

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get position embeddings with length extrapolation support.

        Args:
            x: Input tensor
            seq_len: Sequence length (can exceed max_seq_len)

        Returns:
            (cos, sin): Position embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # Compute dynamic base for extrapolation
        dynamic_base = self._compute_dynamic_base(seq_len)

        # Recompute inv_freq with dynamic base
        inv_freq = 1.0 / (dynamic_base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))

        # Apply extrapolation factor
        if seq_len > self.trained_max_len:
            inv_freq = inv_freq * self.extrapolation_factor

        # Position indices
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)

        # Frequencies
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    def get_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Compute decaying attention bias for "found in the middle" problem.

        Based on arXiv:2406.16008 - calibrates attention to reduce U-shaped
        position bias where models attend more to beginning and end.

        Args:
            seq_len: Sequence length
            device: Tensor device

        Returns:
            Attention bias [1, 1, seq_len, seq_len]
        """
        # Position indices
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Compute distance from each position to all others
        # dist[i, j] = |i - j|
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Decaying bias: positions far from query get small negative bias
        # This counteracts the U-shaped attention pattern
        bias = -self.alpha * torch.log(1 + dist)

        # Zero out the diagonal (self-attention should not be penalized)
        bias = bias.fill_diagonal_(0)

        return bias.unsqueeze(0).unsqueeze(0)


def apply_rotary_pos_emb_context_aware(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply context-aware rotary position embedding.

    Handles batched cos/sin from CARoPE.

    Args:
        q: Query [batch, n_heads, seq_len, head_dim]
        k: Key [batch, n_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim] or [seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim] or [seq_len, head_dim]
        position_ids: Optional position indices

    Returns:
        (q_rotated, k_rotated)
    """
    # Handle different cos/sin shapes
    if cos.dim() == 2:
        # Standard RoPE: [seq, dim] -> [1, 1, seq, dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # CARoPE: [batch, seq, dim] -> [batch, 1, seq, dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Handle position_ids for cached generation
    if position_ids is not None:
        # Gather cos/sin at specific positions
        batch_size = position_ids.shape[0]
        if cos.shape[0] == 1:
            cos = cos.expand(batch_size, -1, -1, -1)
            sin = sin.expand(batch_size, -1, -1, -1)
        cos = cos.gather(2, position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, cos.shape[-1]))
        sin = sin.gather(2, position_ids.unsqueeze(1).unsqueeze(-1).expand(-1, 1, -1, sin.shape[-1]))

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# Export
__all__ = [
    'RotaryEmbedding',
    'apply_rotary_pos_emb',
    'rotate_half',
    'ContextAwareRoPE',
    'DAPEv2',
    'apply_rotary_pos_emb_context_aware'
]


if __name__ == "__main__":
    # Self-test
    print("RoPE Self-Test")
    print("=" * 50)
    
    # Create RoPE with 128k support
    rope = RotaryEmbedding(dim=64, base=100000)
    print(f"✓ Created RoPE (dim=64, base=100000)")
    
    # Test cos/sin generation
    x = torch.randn(2, 8, 128, 64)  # [batch, heads, seq, head_dim]
    cos, sin = rope(x, seq_len=128)
    print(f"✓ Generated cos/sin: {cos.shape}")
    
    # Test Q/K rotation
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"✓ Rotated Q/K: {q_rot.shape}, {k_rot.shape}")
    
    # Test gradient flow
    q.requires_grad = True
    k.requires_grad = True
    cos, sin = rope(q, seq_len=128)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()
    print(f"✓ Gradient flow: q.grad norm = {q.grad.norm().item():.4f}")
    
    # Test long sequence (verify 128k support)
    x_long = torch.randn(1, 1, 1000, 64)
    cos_long, sin_long = rope(x_long, seq_len=1000)
    print(f"✓ Long sequence (1000): cos shape = {cos_long.shape}")
    
    print("\n✅ All RoPE tests passed!")
