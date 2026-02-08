"""
Elleci v2.0 - Configuration System

Centralized configuration using dataclasses for easy experimentation.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class BitNetConfig:
    """Configuration for BitNet 1.58b layers"""
    eps: float = 1e-5
    use_bias: bool = False
    
    def __post_init__(self):
        assert self.eps > 0, "eps must be positive"


@dataclass
class MLAConfig:
    """Configuration for Multi-Head Latent Attention"""
    d_model: int = 768
    n_heads: int = 12
    kv_lora_rank: int = 128  # Compression rank for KV cache
    rope_base: int = 100000  # RoPE base (100000 = 128k context support)
    dropout: float = 0.0
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads


@dataclass
class MambaConfig:
    """Configuration for Mamba State Space Model (v1 and v2)"""
    d_model: int = 768
    d_state: int = 16        # Full state size for 24GB VRAM
    d_conv: int = 4          # Local convolution width
    expand: int = 2          # Expansion factor
    dt_rank: str = "auto"    # Rank of Δ (auto = ceil(d_model/16))
    # Mamba-2 specific
    n_heads: int = 8         # Number of heads for Mamba-2 SSD
    chunk_size: int = 64     # Chunk size for efficient processing
    use_mamba2: bool = True  # Use Mamba-2 instead of Mamba v1
    use_matmul_ssd: bool = True # Use Matmul-based SSD (Scan-free)
    use_checkpointing: bool = False # VRAM optimization (Gradient Checkpointing)
    use_cuda_ssd: bool = True   # Use Custom CUDA Kernel (Max VRAM Savings)

    
    def __post_init__(self):
        if self.dt_rank == "auto":
            self.dt_rank = max(1, self.d_model // 16)


@dataclass
class RouterConfig:
    """Configuration for Adaptive Router"""
    d_model: int = 768
    n_experts: int = 2       # Fast path vs Slow path
    router_hidden: int = 256  # Hidden size of gating network
    temperature: float = 1.0  # Softmax temperature
    
    
@dataclass
class ThinkingLoopConfig:
    """Configuration for recurrent thinking loop (supports PonderNet)"""
    max_iterations: int = 4   # Maximum thinking iterations
    convergence_threshold: float = 0.01  # Stop if change < threshold (legacy)
    # PonderNet parameters
    use_pondernet: bool = True  # Use learned halting instead of convergence
    lambda_p: float = 0.5       # Geometric prior for halting (higher = halt sooner)
    kl_weight: float = 0.01     # Weight for KL regularization loss
    epsilon: float = 0.05       # Minimum probability floor to avoid collapse


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts (Elleci v2)

    Reduced config for A100 40GB: 4 experts, top-1 routing (~2.8B params)
    Original was: 8 experts, top-2 routing (~5.84B params)
    """
    d_model: int = 2560
    num_experts: int = 8       # Increased to 8 for 7B capacity
    top_k: int = 1             # Top-1 efficiency
    # MoE layers: Apply to Mamba layers (0, 1, 2, 4, 5, 6... excluding MLA at 3, 7, 11...)
    # For 32 layers: MLA at 3, 7, 11, 15, 19, 23, 27, 31
    moe_layers: tuple = tuple(i for i in range(32) if i % 4 != 3) 
    # Drop-Upcycling
    drop_ratio: float = 0.2
    # Router settings
    router_type: str = "top_k"  # "top_k" or "expert_choice"
    router_jitter: float = 0.0  # Noise for load balancing during training
    # Auxiliary losses
    aux_loss_weight: float = 0.01
    dpsl_prior: float = 0.25   # 1/4 for 4 experts (uniform)
    spr_loss_weight: float = 0.1  # Weight for Similarity Preserving Router Loss (ArXiv 2506.14038)
    max_spr_tokens: int = 256  # Max tokens for SPR computation (avoids OOM)
    erc_loss_weight: float = 0.1  # Weight for Expert-Router Coupling Loss (ArXiv 2512.23447)
    # MoE++ settings
    zero_computation_experts: bool = False
    # FFN expansion
    ffn_expansion: float = 2.0  # Reduced from 8/3 (2.67) to save params
    quantization_bit: Optional[int] = None # BitNet quantization (e.g. 2 for 1.58-bit)

    def __post_init__(self):
        if self.dpsl_prior is None:
            self.dpsl_prior = 1.0 / self.num_experts
    

@dataclass
class TrainingConfigV2:
    """Configuration for Elleci v2 3-phase training"""
    # Total steps per phase
    phase1_steps: int = 35000  # English Foundation (60%)
    phase2_steps: int = 15000  # Italian Knowledge (25%)
    phase3_steps: int = 10000  # Instruction Alignment (15%)

    # Curriculum learning: (progress_threshold, seq_len)
    curriculum_schedule: tuple = (
        (0.15, 1024),  # First 15%: 1024 tokens
        (0.40, 2048),  # 15-40%: 2048 tokens
        (0.70, 4096),  # 40-70%: 4096 tokens
        (1.00, 8192),  # 70-100%: 8192 tokens (Full Context)
    )

    # Instruction phase (optimized for A100 40/80GB)
    instruction_curriculum: tuple = (
        (0.40, 4096),  # First 40%: 4096 tokens
        (1.00, 8192),  # Last 60%: 8192 tokens
    )

    # Warmup
    warmup_ratio: float = 0.05  # 5% warmup

    # SWA (Stochastic Weight Averaging)
    swa_start_ratio: float = 0.80  # Start at 80% of training
    swa_lr_factor: float = 0.5

    # Checkpointing
    save_interval: int = 5000
    eval_interval: int = 1000
    keep_last_checkpoints: int = 2  # Ridotto per risparmiare storage su Vast.ai

    @property
    def total_steps(self) -> int:
        return self.phase1_steps + self.phase2_steps + self.phase3_steps


@dataclass
class ElleciConfig:
    """Main model configuration"""
    # Model architecture
    d_model: int = 2560  # Elleci-7B (Optimized for A100 80GB)
    n_layers: int = 32   # Deeper reasoning
    vocab_size: int = 32128
    max_seq_len: int = 8192   # Extended context (A100 80GB support)

    # Sub-module configs
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    thinking: ThinkingLoopConfig = field(default_factory=ThinkingLoopConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    training_v2: TrainingConfigV2 = field(default_factory=TrainingConfigV2)

    # Architecture flags
    use_router: bool = False  # Default to False (Production v2 backbone)
    use_moe: bool = True     # Enable Mixture of Experts (Elleci v2-MoE)
    use_v2: bool = True       # Enable v2 modules (DiffMamba2, EG-MLA) - ENABLED for better performance
    use_4bit_act: bool = False  # Enable 4-bit activations (BitNet a4.8)
    ffn_type: str = "swiglu"  # Standard V2-A
    norm_type: str = "rms"    # Standard V2-A

    # Dropout (regularization)
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4  # Updated for v2
    weight_decay: float = 0.1    # Updated for v2
    max_grad_norm: float = 1.0

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: str = "bf16"  # bf16, fp16, or none
    seed: int = 42

    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    def __post_init__(self):
        # Ensure sub-configs have correct d_model
        self.mla.d_model = self.d_model
        self.mamba.d_model = self.d_model
        self.router.d_model = self.d_model
        self.moe.d_model = self.d_model

        # Auto-set n_heads based on d_model (head_dim should be 64-128)
        # d_model=768 -> 12 heads (head_dim=64)
        # d_model=1024 -> 16 heads (head_dim=64)
        # d_model=2048 -> 16 heads (head_dim=128)
        if self.d_model >= 2048:
            self.mla.n_heads = 16  # head_dim = 128
        elif self.d_model >= 1024:
            self.mla.n_heads = 16  # head_dim = 64
        elif self.d_model >= 768:
            self.mla.n_heads = 12  # head_dim = 64 for 768d
        else:
            # For smaller models, find largest divisor <= 16
            for n in [16, 12, 8, 4, 2, 1]:
                if self.d_model % n == 0:
                    self.mla.n_heads = n
                    break

        # Verify divisibility
        assert self.d_model % self.mla.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.mla.n_heads})"
        
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }


if __name__ == "__main__":
    # Test config creation
    config = ElleciConfig()
    print("Elleci Configuration:")
    print(f"  Model size: {config.d_model}d × {config.n_layers} layers")
    print(f"  Vocab: {config.vocab_size}")
    print(f"  Device: {config.device}")
    print(f"  MLA heads: {config.mla.n_heads}")
    print(f"  Mamba state: {config.mamba.d_state}")
    print("✓ Config validated")
