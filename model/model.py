"""
Elleci v2.0 - Complete Integrated Model

Hybrid architecture combining:
- BitNet 1.58b (ternary weights)
- MLA (KV cache compression)
- Mamba (state space model)
- Adaptive Router (Fast/Slow paths)
- Thinking Loop (recurrent reasoning)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from .common.bitnet_legacy import BitLinear, BitLinear_a4
try:
    from .common.bitnet_triton import BitNetLinear as BitNetLinearTriton
    TRITON_BITNET_AVAILABLE = True
    print("🚀 Triton BitNet kernel available")
except ImportError:
    TRITON_BITNET_AVAILABLE = False
    print("⚠️ Triton BitNet kernel NOT available - falling back to legacy PyTorch implementation")

from .common.mla import MLASelfAttention, EGMLASelfAttention
from .common.mamba_legacy import MambaBlock

# Core Mamba Optimizations
from core.mamba.mamba2 import Mamba2Block as Mamba2BlockFast, DifferentialMamba2Block
from core.mamba.mamba2_cuda import DifferentialMamba2BlockCuda
from core.mamba.mamba2_checkpointed import DifferentialMamba2BlockCheckpoint
from core.mamba.mamba2_optimized import DifferentialMamba2BlockMatmul

from .common.router import AdaptiveRouter
from .common.thinking_loop import ThinkingLoop, PonderNetThinkingLoop
from .common.moe import MoEFFN, MoEConfig
from .common.rope import ContextAwareRoPE

# Liger Kernel for fused operations (memory + speed optimization)
# Bug fixed in liger-kernel 0.6.4+ - Fused CE now works with vocab 32128 + bf16
LIGER_AVAILABLE = False
LIGER_RMSNORM_AVAILABLE = False
LIGER_FUSED_CE_AVAILABLE = False
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    LIGER_AVAILABLE = True
    LIGER_FUSED_CE_AVAILABLE = True
    print("🐯 Liger Fused CE available (saves ~0.5-1GB VRAM)")
except ImportError:
    pass

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    LIGER_RMSNORM_AVAILABLE = True
    print("🐯 Liger RMSNorm available (fused kernel for ~0.3GB VRAM savings)")
except ImportError:
    pass


# RMSNorm Implementation - uses Liger kernel if available
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_liger = LIGER_RMSNORM_AVAILABLE

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_liger:
            # Use Liger's fused kernel for better memory efficiency
            return LigerRMSNormFunction.apply(x, self.weight, self.eps)
        else:
            # Fallback to manual implementation
            # Cast to float32 for numerical stability, then back to input dtype
            output = self._norm(x.float()).type_as(x)
            # Ensure weight dtype matches to avoid mismatch warning
            return output * self.weight.type_as(output)

# SwiGLU Feed-Forward Network
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, expansion_factor=8/3, use_4bit_act=False):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        # Ensure hidden_dim is multiple of 256 for efficiency
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)

        # v2: Use 4-bit activations for better efficiency
        if use_4bit_act:
            LinearClass = BitLinear_a4
        else:
            # Use Triton kernel if available and 4-bit activations are NOT requested
            # (Triton kernel currently only implements 1.58-bit weights + BF16 activations)
            LinearClass = BitNetLinearTriton if TRITON_BITNET_AVAILABLE else BitLinear

        self.w1 = LinearClass(d_model, hidden_dim)  # Gate projection
        self.w2 = LinearClass(d_model, hidden_dim)  # Up projection
        self.w3 = LinearClass(hidden_dim, d_model)  # Down projection

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ElleciBlock(nn.Module):
    """
    Single transformer block with hybrid Mamba/MLA attention.

    v2 Features:
    - DifferentialMamba2Block for Mamba layers (better context focus)
    - EGMLASelfAttention for MLA layers (59.9% extra KV compression)
    - MoE-FFN for Mamba layers (4B total, 1.2B active)
    - BitLinear_a4 for 4-bit activations (55% sparsity)

    Can use either MLA or Mamba based on configuration.
    """
    def __init__(self, config, use_mamba=False, layer_idx=0):
        super().__init__()
        self.use_mamba = use_mamba
        self.layer_idx = layer_idx

        # v2 flags
        use_v2 = getattr(config, 'use_v2', False)
        use_moe = getattr(config, 'use_moe', False)
        use_4bit = getattr(config, 'use_4bit_act', False)

        # Dropout for regularization (prevents overfitting)
        dropout_rate = getattr(config, 'dropout', 0.1)
        self.dropout = nn.Dropout(dropout_rate)

        # Attention mechanism
        if use_mamba:
            # Use Mamba-2 if enabled in config, else fallback to v1
            if getattr(config.mamba, 'use_mamba2', True):
                # v2: Use Differential Mamba for better context handling
                if use_v2:
                    # Check for Matmul Optimization (Phase 5)
                    if config.mamba.use_cuda_ssd:
                        # 1. Custom CUDA Kernel (Best VRAM)
                        self.attn = DifferentialMamba2BlockCuda(config.mamba)
                    elif config.mamba.use_checkpointing:
                        # 2. PyTorch Checkpointing (Good VRAM)
                        self.attn = DifferentialMamba2BlockCheckpoint(config.mamba)
                    elif config.mamba.use_matmul_ssd:
                        # 3. Matmul Pure PyTorch (Fast but Hungry)
                        self.attn = DifferentialMamba2BlockMatmul(config.mamba)
                    else:
                        self.attn = DifferentialMamba2Block(config.mamba)
                else:
                    self.attn = Mamba2BlockFast(config.mamba)
            else:
                self.attn = MambaBlock(config.mamba)
        else:
            # v2: Use EG-MLA for extra KV compression
            if use_v2:
                self.attn = EGMLASelfAttention(config.mla)
            else:
                self.attn = MLASelfAttention(config.mla)

        # FFN
        # v2: MoE for Mamba layers, dense for MLA layers
        ffn_type = getattr(config, 'ffn_type', 'swiglu')

        if use_moe and use_mamba:
            # MoE-FFN for Mamba layers (even layers)
            moe_layers = list(getattr(config.moe, 'moe_layers', []))
            if layer_idx in moe_layers:
                self.ffn = MoEFFN(config.d_model, config.moe)
                self.is_moe = True
            else:
                self.ffn = SwiGLUFFN(config.d_model, use_4bit_act=use_4bit)
                self.is_moe = False
        elif ffn_type == 'swiglu':
            self.ffn = SwiGLUFFN(config.d_model, use_4bit_act=use_4bit)
            self.is_moe = False
        else:  # Legacy GELU
            # Use Triton if available, else legacy
            LinearClass = BitNetLinearTriton if TRITON_BITNET_AVAILABLE else BitLinear
            self.ffn = nn.Sequential(
                LinearClass(config.d_model, config.d_model * 4),
                nn.GELU(),
                LinearClass(config.d_model * 4, config.d_model),
            )
            self.is_moe = False

        # Norm (V2: RMSNorm default)
        norm_type = getattr(config, 'norm_type', 'rms')
        NormClass = RMSNorm if norm_type == 'rms' else nn.LayerNorm
        self.norm1 = NormClass(config.d_model)
        self.norm2 = NormClass(config.d_model)

    def forward(self, x, use_cache=False, past_kv=None, past_mamba_state=None):
        """
        Forward pass through block.

        Args:
            x: Input tensor [batch, seq, d_model]
            use_cache: Whether to return KV/Mamba state cache
            past_kv: Previous KV cache (only for MLA blocks)
            past_mamba_state: Previous Mamba SSM state (only for Mamba blocks)

        Returns:
            If use_cache: (output, present_kv, present_mamba_state)
            Else: output
        """
        present_kv = None
        present_mamba_state = None

        # Attention + dropout + residual
        if use_cache:
            if self.use_mamba:
                # Mamba with state caching
                attn_out, present_mamba_state = self.attn(
                    self.norm1(x), use_cache=True, past_state=past_mamba_state
                )
                x = x + self.dropout(attn_out)
            else:
                # MLA/EGMLA with KV caching
                attn_out, present_kv = self.attn(
                    self.norm1(x), use_cache=True, past_kv=past_kv
                )
                x = x + self.dropout(attn_out)
        else:
            # No caching
            x = x + self.dropout(self.attn(self.norm1(x)))

        # FFN + dropout + residual
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if use_cache:
            return x, present_kv, present_mamba_state
        return x

    def get_aux_loss(self):
        """Get MoE auxiliary loss if applicable."""
        if self.is_moe and hasattr(self.ffn, 'aux_loss'):
            return self.ffn.aux_loss
        return None


class Elleci(nn.Module):
    """
    Complete Elleci model with adaptive routing.
    
    Architecture:
    - Input embedding
    - Router decides path per input
    - Fast path: 2 shallow blocks
    - Slow path: Thinking loop (4 iterations)
    - Output head
    
    Args:
        config: ElleciConfig
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings (RoPE is handled by MLA, no absolute pos_emb needed)
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        
        
        # Main blocks (shared)
        # Pattern 75/25: 3 Mamba, 1 MLA (layers 0,1,2=Mamba, 3=MLA, 4,5,6=Mamba, 7=MLA, ...)
        # This reduces KV cache by 50% (6 MLA instead of 12 for 24 layers)
        self.blocks = nn.ModuleList([
            ElleciBlock(config, use_mamba=(i % 4 != 3), layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Router (Optional)
        if config.use_router:
            self.router = AdaptiveRouter(config.router)

            # Fast path: 2 lightweight blocks
            self.fast_blocks = nn.ModuleList([
                ElleciBlock(config, use_mamba=False) for _ in range(2)
            ])
        else:
            self.router = None
            self.fast_blocks = None

        # Thinking Loop (System 2)
        # Shared Weights: Use the last block (MLA) for reasoning
        # This ensures the thinking loop uses trained weights without extra parameters
        if getattr(config.thinking, 'use_pondernet', False):
            # PonderNet: learned halting with KL regularization
            self.thinking_loop = PonderNetThinkingLoop(
                config.thinking, self.blocks[-1], config.d_model
            )
            self.use_pondernet = True
        else:
            # Legacy: convergence-based halting
            self.thinking_loop = ThinkingLoop(config.thinking, self.blocks[-1])
            self.use_pondernet = False
        
        # Output
        norm_type = getattr(config, 'norm_type', 'rms')
        if norm_type == 'rms':
            self.norm_f = RMSNorm(config.d_model)
        else:
            self.norm_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights (optional)
        # self.lm_head.weight = self.token_emb.weight
        
        # Initialize
        # Initialize
        self.apply(self._init_weights)

        # Gradient Checkpointing flag
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.
        """
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.
        """
        self.gradient_checkpointing = False
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        """
        Forward pass.
        
        Args:
            idx: Token indices [batch, seq_len]
            targets: Target tokens [batch, seq_len] (for training)
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: (Optional) cross-entropy loss
        """
        batch_size, seq_len = idx.shape
        
        # Embeddings (RoPE provides position info in attention layers)
        x = self.token_emb(idx)  # [batch, seq, d_model]
        
        # Router decision (if enabled)
        load_balance_loss = None
        ponder_loss = None
        
        if self.config.use_router and self.router is not None:
            if self.training:
                # ===== TOKEN-LEVEL SOFT ROUTING (CRITICAL FIX) =====
                # Get routing probabilities PER TOKEN (causal!)
                route_probs = self.router(x)  # [batch, seq, 2]
                
                # Extract slow path probability for each token
                p_slow = route_probs[:, :, 1].unsqueeze(-1)  # [batch, seq, 1]
                p_fast = 1.0 - p_slow  # [batch, seq, 1]
                
                # Load balancing loss (aggregate over all tokens)
                target_slow_ratio = 0.3
                actual_slow_ratio = route_probs[:, :, 1].mean()  # Mean over batch AND sequence!
                load_balance_loss = 0.01 * (actual_slow_ratio - target_slow_ratio).pow(2)
                
                # ===== SOFT MIXING: Compute BOTH paths =====
                # Fast path
                x_fast = x.clone()
                for block in self.fast_blocks:
                    x_fast = block(x_fast)
                
                # Slow path (thinking loop processes whole sequence)
                if self.use_pondernet:
                    x_slow, _, ponder_loss = self.thinking_loop(x)
                else:
                    x_slow, _ = self.thinking_loop(x)
                
                # Weighted sum (DIFFERENTIABLE!)
                # Gradient flows to router based on which path helped more
                x = (x_slow * p_slow) + (x_fast * p_fast)  # [batch, seq, d_model]
            else:
                # ===== INFERENCE: Optional hard routing for speed =====
                # For now, use soft mixing for stability
                route_probs = self.router(x)
                p_slow = route_probs[:, :, 1].unsqueeze(-1)
                p_fast = 1.0 - p_slow
                
                x_fast = x.clone()
                for block in self.fast_blocks:
                    x_fast = block(x_fast)
                if self.use_pondernet:
                    x_slow, _, _ = self.thinking_loop(x)  # No loss at inference
                else:
                    x_slow, _ = self.thinking_loop(x)
                
                x = (x_slow * p_slow) + (x_fast * p_fast)
        else:
            # Default: use main blocks without routing
            # Default: use main blocks without routing
            for block in self.blocks:
                if self.gradient_checkpointing and self.training:
                    # Use checkpointing to save VRAM (trades compute for memory)
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
        
        # Output normalization
        x = self.norm_f(x)
        
        # Compute loss if targets provided
        loss = None
        logits = None
        
        if targets is not None:
            if LIGER_FUSED_CE_AVAILABLE and self.training:
                # Liger Fused Linear Cross-Entropy: fuses lm_head + CE for VRAM savings
                # Shift hidden states for causal LM (predict next token)
                shift_hidden = x[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                # Flatten for batch processing
                batch_seq = shift_hidden.size(0) * shift_hidden.size(1)
                shift_hidden_flat = shift_hidden.view(batch_seq, -1)
                shift_labels_flat = shift_labels.view(-1)
                
                # Liger handles linear projection + cross entropy in one fused kernel
                fused_ce = LigerFusedLinearCrossEntropyLoss(
                    ignore_index=-100,
                    label_smoothing=0.1
                )
                main_loss = fused_ce(
                    self.lm_head.weight,  # Weight only, Liger does the matmul
                    shift_hidden_flat,
                    shift_labels_flat
                )
                
                # Also compute logits for logging (only if needed)
                logits = self.lm_head(x)
            else:
                # Standard path: compute logits then cross entropy
                logits = self.lm_head(x)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = targets[..., 1:].contiguous()
                
                main_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.1
                )
            
            # Add load balancing loss if router was used
            if load_balance_loss is not None:
                loss = main_loss + load_balance_loss
            else:
                loss = main_loss
            
            # Add PonderNet loss if enabled
            if ponder_loss is not None:
                loss = loss + ponder_loss
        else:
            logits = self.lm_head(x)  # Only compute logits if no targets (inference)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True,
                 thinking_mode=False, thinking_iterations=None):
        """
        Generate tokens autoregressively with KV + Mamba state caching.

        Args:
            idx: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (if not None)
            use_cache: Whether to use caching for faster generation
            thinking_mode: If True, apply thinking loop to prompt before generation (System 2)
            thinking_iterations: Override max iterations for thinking loop (default: config value)

        Returns:
            Generated tokens [batch, seq_len + max_new_tokens]
            If thinking_mode=True, also returns thinking_info dict with iterations used
        """
        thinking_info = None

        if not use_cache:
            # Original slow path (for debugging)
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

        # Fast path with KV + Mamba state caching
        batch_size = idx.size(0)

        # Initialize caches for each block
        past_kvs = [None] * len(self.blocks)  # KV cache for MLA blocks
        past_mamba_states = [None] * len(self.blocks)  # SSM state for Mamba blocks

        # Process initial prompt (prefill)
        x = self.token_emb(idx)

        # Thinking Mode (System 2): Apply iterative refinement before generation
        if thinking_mode:
            x_original = x.clone()  # Keep original for residual
            # PonderNet returns 3 values, legacy ThinkingLoop returns 2
            if self.use_pondernet:
                x_thinking, n_iterations, _ = self.thinking_loop(x, n_iterations=thinking_iterations)
            else:
                x_thinking, n_iterations = self.thinking_loop(x, n_iterations=thinking_iterations)
            # Residual connection: blend thinking output with original
            x = x + x_thinking
            thinking_info = {
                'iterations_used': n_iterations,
                'max_iterations': self.thinking_loop.max_iterations,
                'converged': n_iterations < self.thinking_loop.max_iterations,
            }

        for i, block in enumerate(self.blocks):
            x, past_kvs[i], past_mamba_states[i] = block(
                x, use_cache=True, past_kv=None, past_mamba_state=None
            )

        x = self.norm_f(x)
        logits = self.lm_head(x)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # Generate remaining tokens one at a time with caching
        for _ in range(max_new_tokens - 1):
            # Check sequence length limit
            if idx.size(1) > self.config.max_seq_len:
                # Truncate and invalidate cache
                idx = idx[:, -self.config.max_seq_len:]
                past_kvs = [None] * len(self.blocks)
                past_mamba_states = [None] * len(self.blocks)
                # Re-process truncated sequence
                x = self.token_emb(idx)
                for i, block in enumerate(self.blocks):
                    x, past_kvs[i], past_mamba_states[i] = block(
                        x, use_cache=True, past_kv=None, past_mamba_state=None
                    )
            else:
                # Process only the new token with cached states
                x = self.token_emb(idx_next)
                for i, block in enumerate(self.blocks):
                    x, past_kvs[i], past_mamba_states[i] = block(
                        x, use_cache=True,
                        past_kv=past_kvs[i],
                        past_mamba_state=past_mamba_states[i]
                    )

            x = self.norm_f(x)
            logits = self.lm_head(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        if thinking_mode:
            return idx, thinking_info
        return idx



# Export
__all__ = ['Elleci', 'ElleciBlock']


if __name__ == "__main__":
    # Self-test
    print("Elleci Model Self-Test")
    print("=" * 60)
    
    import sys
    sys.path.insert(0, '..')
    from config import ElleciConfig
    
    config = ElleciConfig()
    config.n_layers = 4  # Smaller for testing
    config.max_seq_len = 64
    
    print(f"Config: {config.n_layers} layers, d_model={config.d_model}")
    
    model = Elleci(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params/1e6:.1f}M parameters")
    
    # Test forward
    idx = torch.randint(0, config.vocab_size, (2, 32))  # [batch=2, seq=32]
    logits, _ = model(idx) # use_router is handled by config
    print(f"✓ Forward pass: {idx.shape} → {logits.shape}")
    
    # Test with loss
    targets = torch.randint(0, config.vocab_size, (2, 32))
    logits, loss = model(idx, targets=targets)
    print(f"✓ With targets: loss={loss.item():.4f}")
    
    # Test generation
    start_tokens = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(start_tokens, max_new_tokens=10)
    print(f"✓ Generation: {start_tokens.shape} → {generated.shape}")
    
    # Test gradient flow
    loss.backward()
    print(f"✓ Gradient flow: token_emb grad norm = {model.token_emb.weight.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
    print("\n🎉 Elleci v2.0 architecture complete!")
