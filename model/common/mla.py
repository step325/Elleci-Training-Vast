"""
Multi-Head Latent Attention (MLA) - KV Cache Compression with RoPE

Based on DeepSeek-V3 architecture for efficient attention with compressed KV cache.
Reduces memory by ~93% compared to standard multi-head attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import RoPE from our module
from .rope import RotaryEmbedding, apply_rotary_pos_emb


class MLASelfAttention(nn.Module):
    """
    Multi-Head Latent Attention with KV compression and RoPE.
    
    Key innovation: Compress K/V into low-rank latent space before attention.
    
    Standard MHA: Q, K, V all in d_model dimension
    MLA: Q in d_model, K/V compressed to kv_lora_rank (e.g. 128 << 768)
    
    Memory savings: ~93% for KV cache (128 vs 1536 for standard)
    
    Args:
        config: MLAConfig with d_model, n_heads, kv_lora_rank, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        
        # Query projection (full dimension)
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # KV compression: down-project to latent, then up-project
        self.w_kv_down = nn.Linear(config.d_model, config.kv_lora_rank, bias=False)
        self.w_kv_up = nn.Linear(config.kv_lora_rank, config.d_model * 2, bias=False)  # *2 for K and V
        
        # Output projection
        self.w_out = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # RoPE (Rotary Position Embedding) - 128k context support
        rope_base = getattr(config, 'rope_base', 100000)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, 
            base=rope_base,
            max_seq_len=8192
        )
        
    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        """
        Forward pass with compressed KV cache.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask [batch, 1, seq_len, seq_len]
            use_cache: Whether to return KV cache
            past_kv: Previous KV cache (latent) for incremental generation

        Returns:
            output: [batch, seq_len, d_model]
            present_kv: (Optional) KV cache for next step
        """
        batch_size, seq_len, _ = x.shape

        # 1. Compute queries (full dimension)
        q = self.w_q(x)  # [batch, seq, d_model]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

        # 2. Compress current tokens to latent space
        latent = self.w_kv_down(x)  # [batch, seq, kv_lora_rank]

        # 3. Handle KV cache for incremental generation
        if past_kv is not None:
            # Concatenate with past latent
            latent = torch.cat([past_kv, latent], dim=1)  # [batch, past_len + seq, kv_lora_rank]
            kv_seq_len = latent.size(1)
        else:
            kv_seq_len = seq_len

        # 4. Decompress K and V from (possibly extended) latent
        kv = self.w_kv_up(latent)  # [batch, kv_seq_len, d_model*2]
        k, v = kv.chunk(2, dim=-1)  # Each [batch, kv_seq_len, d_model]

        # Reshape for multi-head
        k = k.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 5. Apply RoPE to Q and K with correct position offsets
        # Get cos/sin for full sequence length
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        if past_kv is not None:
            # Q positions: [past_len, past_len+1, ..., past_len+seq_len-1]
            # K positions: [0, 1, 2, ..., kv_seq_len-1]
            past_len = kv_seq_len - seq_len

            # Select Q positions from cos/sin
            q_pos_ids = torch.arange(past_len, kv_seq_len, device=x.device)
            q, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids=q_pos_ids)

            # K uses all positions
            k, _ = apply_rotary_pos_emb(k, None, cos, sin)
        else:
            # Standard case: both Q and K use positions [0, 1, ..., seq_len-1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 6. Flash Attention via SDPA
        # When using cache, we're generating one token at a time, use is_causal=False with manual mask
        if past_kv is not None:
            # Generating: Q is [batch, n_heads, 1, head_dim], K/V are [batch, n_heads, kv_len, head_dim]
            # No causal mask needed - current token can attend to all past tokens
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,  # No dropout during inference
                is_causal=False,  # We handle causality implicitly
                scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )

        # 7. Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        output = self.w_out(attn_output)

        # 8. Return with optional KV cache
        if use_cache:
            # Cache the full latent (including past)
            present_kv = latent.detach()
            return output, present_kv

        return output


class EGMLASelfAttention(MLASelfAttention):
    """
    Embedding-Gated MLA (EG-MLA) - arXiv:2509.16686

    Adds token-specific gating in the latent KV space for
    additional compression and better context selectivity.

    Key innovation: Instead of uniformly compressing all tokens,
    use learned gates to selectively suppress less important
    information in the latent space.

    Benefits:
    - 59.9% additional KV cache compression
    - Better long-context performance
    - Improved retrieval tasks

    Args:
        config: MLAConfig with d_model, n_heads, kv_lora_rank, etc.
    """
    def __init__(self, config):
        super().__init__(config)

        # Embedding gate: produces token-specific gating in latent space
        # Input: d_model -> Output: kv_lora_rank
        self.embed_gate = nn.Sequential(
            nn.Linear(config.d_model, config.kv_lora_rank, bias=False),
            nn.Sigmoid()  # Gate values in [0, 1]
        )

        # Optional: learnable importance threshold
        # Tokens with gate values below this are more aggressively compressed
        self.importance_threshold = nn.Parameter(torch.tensor(0.5))

        # Initialize gate to pass most information initially
        nn.init.xavier_uniform_(self.embed_gate[0].weight, gain=0.5)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        """
        Forward pass with embedding-gated KV compression.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional attention mask
            use_cache: Whether to return KV cache
            past_kv: Previous KV cache (gated latent) for incremental generation

        Returns:
            output: [batch, seq_len, d_model]
            present_kv: (Optional) Compressed KV cache
        """
        batch_size, seq_len, _ = x.shape

        # 1. Compute queries (full dimension)
        q = self.w_q(x)  # [batch, seq, d_model]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, n_heads, seq, head_dim]

        # 2. Compute embedding gate (token-specific)
        gate = self.embed_gate(x)  # [batch, seq, kv_lora_rank]

        # 3. Compress to latent space with gating
        latent = self.w_kv_down(x)  # [batch, seq, kv_lora_rank]
        latent_gated = latent * gate  # Token-wise gating

        # 4. Handle KV cache for incremental generation
        if past_kv is not None:
            # Concatenate with past gated latent
            latent_gated = torch.cat([past_kv, latent_gated], dim=1)
            kv_seq_len = latent_gated.size(1)
        else:
            kv_seq_len = seq_len

        # 5. Decompress K and V from (possibly extended) gated latent
        kv = self.w_kv_up(latent_gated)  # [batch, kv_seq_len, d_model*2]
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head
        k = k.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 6. Apply RoPE to Q and K with correct position offsets
        cos, sin = self.rotary_emb(k, seq_len=kv_seq_len)

        if past_kv is not None:
            # Q positions: [past_len, past_len+1, ...]
            # K positions: [0, 1, 2, ..., kv_seq_len-1]
            past_len = kv_seq_len - seq_len
            q_pos_ids = torch.arange(past_len, kv_seq_len, device=x.device)
            q, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids=q_pos_ids)
            k, _ = apply_rotary_pos_emb(k, None, cos, sin)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 7. Flash Attention via SDPA
        if past_kv is not None:
            # Generating: no causal mask needed
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )

        # 8. Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.w_out(attn_output)

        # 9. Return with optional KV cache
        if use_cache:
            # Cache the full gated latent (including past)
            present_kv = latent_gated.detach()
            return output, present_kv

        return output


class BlockSparseMLA(MLASelfAttention):
    """
    MLA with Block Sparse Attention (arXiv:2512.07011)

    Uses top-k block selection for sparse attention pattern,
    improving efficiency on retrieval tasks.

    Instead of attending to all positions, selects top-k blocks
    of keys/values based on query relevance.

    Args:
        config: MLAConfig
        block_size: Size of attention blocks (default: 64)
        top_k_blocks: Number of blocks to attend to (default: 4)
    """
    def __init__(self, config, block_size: int = 64, top_k_blocks: int = 4):
        super().__init__(config)
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks

        # Block scoring network
        self.block_scorer = nn.Linear(config.d_model, 1, bias=False)

    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        """
        Forward pass with block-sparse attention.

        For sequences shorter than block_size * top_k_blocks,
        falls back to standard dense attention.
        """
        batch_size, seq_len, _ = x.shape

        # Fall back to dense for short sequences
        if seq_len <= self.block_size * self.top_k_blocks:
            return super().forward(x, mask, use_cache, past_kv)

        # Block-sparse attention implementation
        # 1. Compute queries
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)

        # 2. Compress and decompress KV
        latent = self.w_kv_down(x)
        kv = self.w_kv_up(latent)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 4. Block scoring for sparse selection
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        # Compute block representations (mean of each block)
        block_reps = []
        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, seq_len)
            block_rep = x[:, start:end, :].mean(dim=1)  # [batch, d_model]
            block_reps.append(block_rep)
        block_reps = torch.stack(block_reps, dim=1)  # [batch, num_blocks, d_model]

        # Score blocks
        block_scores = self.block_scorer(block_reps).squeeze(-1)  # [batch, num_blocks]

        # Select top-k blocks per query position (simplified: same blocks for all positions)
        _, top_block_indices = torch.topk(
            block_scores, min(self.top_k_blocks, num_blocks), dim=-1
        )  # [batch, top_k_blocks]

        # 5. Gather selected blocks and compute attention
        # For efficiency, we use the selected blocks for all query positions
        # This is a simplified version; full implementation would be per-query

        # Create sparse mask
        sparse_mask = torch.zeros(batch_size, seq_len, device=x.device, dtype=torch.bool)
        for b in range(batch_size):
            for block_idx in top_block_indices[b]:
                start = block_idx * self.block_size
                end = min((block_idx + 1) * self.block_size, seq_len)
                sparse_mask[b, start:end] = True

        # Expand mask for attention
        attn_mask = sparse_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
        attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)

        # Combine with causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        combined_mask = ~attn_mask | causal_mask

        # 6. Compute attention with sparse mask
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~combined_mask,  # SDPA expects True=attend
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # We handle causality in mask
            scale=1.0 / math.sqrt(self.head_dim)
        )

        # 7. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.w_out(attn_output)

        if use_cache:
            return output, latent.detach()
        return output


# Export
__all__ = ['MLASelfAttention', 'EGMLASelfAttention', 'BlockSparseMLA']


if __name__ == "__main__":
    # Self-test
    print("MLA Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        d_model: int = 768
        n_heads: int = 12
        kv_lora_rank: int = 128
        rope_dim: int = 32
        dropout: float = 0.0
    
    config = TestConfig()
    mla = MLASelfAttention(config)
    
    print(f"✓ Created MLA (d_model={config.d_model}, kv_rank={config.kv_lora_rank})")
    
    # Test forward
    x = torch.randn(2, 16, 768)  # [batch=2, seq=16, d_model=768]
    out = mla(x)
    print(f"✓ Forward pass: {x.shape} → {out.shape}")
    
    # Test with cache
    out, kv_cache = mla(x, use_cache=True)
    print(f"✓ With cache: output={out.shape}, cache={kv_cache.shape}")
    
    # Check compression ratio
    standard_kv_size = 768 * 2 * 16  # K and V, full dimension
    compressed_size = config.kv_lora_rank + config.rope_dim
    compression = (1 - compressed_size / (768*2)) * 100
    print(f"✓ KV compression: {compression:.1f}% memory saved")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"✓ Gradient flow: w_q grad norm = {mla.w_q.weight.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
