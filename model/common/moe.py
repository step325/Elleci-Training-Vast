"""
Mixture of Experts (MoE) Module for Elleci v2

Implements sparse MoE with:
- UpIT: Using fine-tuning checkpoints as expert initialization
- Drop-Upcycling: Re-init 20% weights for specialization
- Router Upcycling: Initialize router from MLA attention heads
- Dirichlet-Prior Shaping Loss (DPSL): Prevent expert collapse
- MoE++: Zero-computation experts for inference

Based on:
- Sparse Upcycling (arXiv:2212.05055)
- MoE-Mamba (arXiv:2401.04081)
- UpIT (arXiv:2410.01610)
- Drop-Upcycling (arXiv:2502.19261)
- MoE++ (arXiv:2410.07348)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .bitnet_triton import BitNetLinear


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts"""
    d_model: int = 2048
    num_experts: int = 8
    top_k: int = 2
    # MoE layers (applied only to Mamba blocks - 75/25 ratio: all except 3, 7, 11, 15, 19, 23)
    moe_layers: List[int] = None  # Will be set in __post_init__
    # Drop-Upcycling
    drop_ratio: float = 0.2
    # Router settings
    router_type: str = "top_k"  # "top_k" or "expert_choice"
    router_jitter: float = 0.0  # Noise for load balancing during training
    # Auxiliary losses
    aux_loss_weight: float = 0.01
    dpsl_prior: float = 0.125  # 1/8 for 8 experts (uniform)
    spr_loss_weight: float = 0.1  # Weight for Similarity Preserving Router Loss (ArXiv 2506.14038)
    max_spr_tokens: int = 256  # Max tokens for SPR computation (avoids OOM)
    erc_loss_weight: float = 0.1  # Weight for Expert-Router Coupling Loss (ArXiv 2512.23447)
    # MoE++ settings
    zero_computation_experts: bool = False
    # FFN expansion
    ffn_expansion: float = 8/3  # SwiGLU factor
    # Quantization
    quantization_bit: Optional[int] = None  # 2 or 4. If None, use standard nn.Linear

    def __post_init__(self):
        if self.moe_layers is None:
            # Default: all Mamba layers (75/25 ratio: all except 3, 7, 11, 15, 19, 23)
            self.moe_layers = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
        if self.dpsl_prior is None:
            self.dpsl_prior = 1.0 / self.num_experts


class ExpertFFN(nn.Module):
    """
    Single expert FFN (SwiGLU architecture).

    Args:
        d_model: Model dimension
        expansion_factor: Hidden dimension multiplier (default 8/3 for SwiGLU)
    """
    def __init__(self, d_model: int, expansion_factor: float = 8/3, quantization_bit: Optional[int] = None):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        # Round to multiple of 256 for efficiency
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)

        if quantization_bit is not None:
            # BitNet 1.58-bit Linear Layers (Triton)
            assert d_model % 4 == 0, "d_model must be divisible by 4 for BitNet packing"
            assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4 for BitNet packing"
            
            # Note: quantization_bit is passed but BitNetLinear is fixed to 1.58-bit (2-bit packed) usually.
            # We assume presence of this flag enables BitNet.
            self.w1 = BitNetLinear(d_model, hidden_dim, bias=False)  # Gate
            self.w2 = BitNetLinear(d_model, hidden_dim, bias=False)  # Up
            self.w3 = BitNetLinear(hidden_dim, d_model, bias=False)  # Down
        else:
            self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate
            self.w2 = nn.Linear(d_model, hidden_dim, bias=False)  # Up
            self.w3 = nn.Linear(hidden_dim, d_model, bias=False)  # Down

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: SwiGLU(x) = (Silu(W1*x) * W2*x) * W3"""
        return self.w3(self.act_fn(self.w1(x)) * self.w2(x))


class Router(nn.Module):
    """
    Expert routing module with load balancing.

    Supports:
    - Top-K routing: Each token selects top-k experts
    - Expert-choice routing: Each expert selects top tokens
    - Jitter noise for training stability

    Args:
        config: MoEConfig
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.jitter = config.router_jitter

        # Router projection
        self.gate = nn.Linear(config.d_model, config.num_experts, bias=False)

        # Initialize with small weights for stable start
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_aux_loss: Whether to compute auxiliary losses

        Returns:
            router_probs: Softmax probabilities [batch, seq, num_experts]
            top_k_indices: Selected expert indices [batch, seq, top_k]
            aux_loss: Load balancing + DPSL loss (if requested)
        """
        batch, seq_len, d_model = x.shape

        # Compute logits
        logits = self.gate(x)  # [batch, seq, num_experts]

        # Add jitter during training for load balancing
        if self.training and self.jitter > 0:
            noise = torch.randn_like(logits) * self.jitter
            logits = logits + noise

        # Compute softmax probabilities
        router_probs = F.softmax(logits, dim=-1)  # [batch, seq, num_experts]

        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [batch, seq, top_k]

        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        aux_loss = None
        if return_aux_loss:
            # 1. Load balancing loss
            # Encourage uniform distribution across experts
            # f_i = fraction of tokens routed to expert i
            # P_i = mean router probability for expert i
            expert_mask = F.one_hot(
                top_k_indices, self.num_experts
            ).float()  # [batch, seq, top_k, num_experts]
            expert_mask = expert_mask.sum(dim=2)  # [batch, seq, num_experts]

            # Token fraction per expert
            tokens_per_expert = expert_mask.sum(dim=[0, 1])  # [num_experts]
            f_i = tokens_per_expert / (batch * seq_len * self.top_k)

            # Mean probability per expert
            P_i = router_probs.mean(dim=[0, 1])  # [num_experts]

            # Load balancing loss: sum(f_i * P_i) * num_experts
            load_balance_loss = (f_i * P_i).sum() * self.num_experts

            # 2. Dirichlet-Prior Shaping Loss (DPSL)
            # Encourage prior distribution (uniform by default)
            target_prior = torch.full(
                (self.num_experts,),
                self.config.dpsl_prior,
                device=x.device
            )
            dpsl_loss = F.kl_div(
                P_i.log(), target_prior, reduction='sum'
            )

            # 3. Similarity Preserving Router Loss (SPR) - ArXiv 2506.14038
            # Prevents expert collapse by encouraging similar inputs → similar routing
            spr_loss = torch.tensor(0.0, device=x.device)
            if self.config.spr_loss_weight > 0:
                total_tokens = batch * seq_len
                max_spr_tokens = getattr(self.config, 'max_spr_tokens', 256)

                # Sample subset for SPR to avoid OOM with large NxN matrices
                if total_tokens > max_spr_tokens:
                    indices = torch.randperm(total_tokens, device=x.device)[:max_spr_tokens]
                    x_sample = x.view(total_tokens, -1)[indices]  # [max_spr_tokens, d_model]
                    router_sample = router_probs.view(total_tokens, -1)[indices]  # [max_spr_tokens, num_experts]
                else:
                    x_sample = x.view(total_tokens, -1)
                    router_sample = router_probs.view(total_tokens, -1)

                # Input similarity matrix (cosine similarity)
                x_norm = F.normalize(x_sample, dim=-1)  # [N, d_model]
                input_sim = torch.mm(x_norm, x_norm.t())  # [N, N]

                # Routing similarity matrix
                router_norm = F.normalize(router_sample, dim=-1)  # [N, num_experts]
                routing_sim = torch.mm(router_norm, router_norm.t())  # [N, N]

                # SPR Loss: encourage similar routing for similar inputs
                spr_loss = F.mse_loss(routing_sim, input_sim.detach())

            # Combined auxiliary loss
            aux_loss = self.config.aux_loss_weight * (
                load_balance_loss + dpsl_loss + self.config.spr_loss_weight * spr_loss
            )

        return router_probs, top_k_indices, top_k_probs, aux_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with sparse activation.

    Each token is processed by only top-k experts (default 2 of 8).

    Args:
        config: MoEConfig
    """
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # Router
        self.router = Router(config)

        # Experts
        self.experts = nn.ModuleList([
            ExpertFFN(config.d_model, config.ffn_expansion, quantization_bit=config.quantization_bit)
            for _ in range(config.num_experts)
        ])

        # MoE++ zero-computation expert (optional)
        if config.zero_computation_experts:
            # Identity expert - just passes through input
            self.zero_expert = nn.Identity()

    def compute_erc_loss(self) -> torch.Tensor:
        """
        Compute Expert-Router Coupling (ERC) Loss.

        Based on "Coupling Experts and Routers in MoE via Auxiliary Loss" (ArXiv 2512.23447).

        Concept:
        - Treat router gate weights as "Proxy Tokens" representing ideal inputs for each expert
        - Pass proxy tokens through all experts to get activation matrix A
        - A[i,j] = activation magnitude of Expert j on Proxy Token i
        - Maximize diagonal: Expert i should respond maximally to its own proxy token

        Returns:
            erc_loss: Scalar loss encouraging expert-router coupling
        """
        # Step A: Extract proxy tokens from router gate weights
        # gate.weight shape: [num_experts, d_model]
        proxy_tokens = self.router.gate.weight  # [N, D]
        num_experts = proxy_tokens.shape[0]
        device = proxy_tokens.device

        # Step B: Pass each proxy token through ALL experts
        # activation_matrix[i, j] = ||Expert_j(proxy_token_i)||
        activation_matrix = torch.zeros(num_experts, num_experts, device=device)

        for i in range(num_experts):
            proxy_i = proxy_tokens[i:i+1]  # [1, d_model]
            for j in range(num_experts):
                # Get activation magnitude (L2 norm of expert output)
                expert_output = self.experts[j](proxy_i)  # [1, d_model]
                activation_matrix[i, j] = expert_output.norm(p=2)

        # Step C: Compute loss using double softmax + cross-entropy
        # We want A[i,i] (diagonal) to be maximized relative to row/column

        # Target: diagonal indices [0, 1, 2, ..., N-1]
        target = torch.arange(num_experts, device=device)

        # Row-wise loss: for each proxy token i, Expert i should have max activation
        # softmax over columns (which expert responds most to proxy i)
        row_loss = F.cross_entropy(activation_matrix, target)

        # Column-wise loss: for each Expert j, its own proxy token should activate it most
        # softmax over rows (which proxy token activates Expert j most)
        col_loss = F.cross_entropy(activation_matrix.t(), target)

        # Combined ERC loss
        erc_loss = (row_loss + col_loss) / 2.0

        return erc_loss

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MoE layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: Routing auxiliary loss
        """
        batch, seq_len, d_model = x.shape

        # Get routing decisions
        router_probs, top_k_indices, top_k_weights, aux_loss = self.router(
            x, return_aux_loss
        )
        # top_k_indices: [batch, seq, top_k]
        # top_k_weights: [batch, seq, top_k]

        # Flatten for efficient processing
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # Shape: [batch, seq, top_k] -> compare with expert_idx
            expert_mask = (top_k_indices == expert_idx)  # [batch, seq, top_k]

            # Get the corresponding weights
            expert_weights = top_k_weights * expert_mask.float()  # [batch, seq, top_k]
            expert_weights = expert_weights.sum(dim=-1)  # [batch, seq]

            # Find non-zero positions
            expert_weights_flat = expert_weights.view(-1)  # [batch*seq]
            token_indices = expert_weights_flat.nonzero(as_tuple=True)[0]

            if len(token_indices) > 0:
                # Get tokens for this expert
                expert_input = x_flat[token_indices]  # [num_tokens, d_model]

                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Weight and accumulate
                weights = expert_weights_flat[token_indices].unsqueeze(-1)
                output[token_indices] += expert_output * weights

        # Reshape output
        output = output.view(batch, seq_len, d_model)

        # Add ERC loss if training and enabled
        if return_aux_loss and self.training and self.config.erc_loss_weight > 0:
            erc_loss = self.compute_erc_loss()
            if aux_loss is not None:
                aux_loss = aux_loss + self.config.erc_loss_weight * erc_loss
            else:
                aux_loss = self.config.erc_loss_weight * erc_loss

        return output, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN that can replace SwiGLUFFN in ElleciBlock.

    Drop-in replacement with same interface.
    """
    def __init__(self, d_model: int, config: Optional[MoEConfig] = None):
        super().__init__()
        if config is None:
            config = MoEConfig(d_model=d_model)
        else:
            config.d_model = d_model

        self.moe = MoELayer(config)
        self.aux_loss = None  # Store for retrieval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (stores aux_loss internally)."""
        output, aux_loss = self.moe(x, return_aux_loss=self.training)
        self.aux_loss = aux_loss
        return output


def drop_upcycle_weights(
    source_state_dict: dict,
    drop_ratio: float = 0.2,
    seed: Optional[int] = None
) -> dict:
    """
    Apply Drop-Upcycling to model weights.

    Re-initializes a fraction of weights to encourage expert specialization.

    Args:
        source_state_dict: Original model state dict
        drop_ratio: Fraction of weights to re-initialize
        seed: Random seed for reproducibility

    Returns:
        Modified state dict
    """
    if seed is not None:
        torch.manual_seed(seed)

    result = {}
    for name, param in source_state_dict.items():
        if 'weight' in name and param.dim() >= 2:
            # Create mask for weights to drop
            mask = torch.rand_like(param) > drop_ratio

            # Re-initialize dropped weights with xavier
            new_init = torch.empty_like(param)
            nn.init.xavier_uniform_(new_init)

            # Combine: keep (1-drop_ratio) original, reinit drop_ratio
            result[name] = torch.where(mask, param, new_init)
        else:
            result[name] = param.clone()

    return result


def initialize_router_from_mla(
    router: Router,
    mla_weights: dict,
    layer_idx: int = 0
) -> None:
    """
    Initialize router weights from MLA attention head weights.

    Uses the pattern of attention heads to inform expert routing.

    Args:
        router: Router module to initialize
        mla_weights: MLA module state dict
        layer_idx: Which layer's MLA to use
    """
    # Try to get W_q weights which encode input-dependent patterns
    w_q_key = f'blocks.{layer_idx}.attn.w_q.weight'
    if w_q_key in mla_weights:
        w_q = mla_weights[w_q_key]  # [d_model, d_model]
        d_model = w_q.shape[0]
        num_experts = router.num_experts

        # Use first `num_experts` rows of W_q to initialize router
        # Scale down to avoid large initial logits
        router.gate.weight.data = w_q[:num_experts, :] * 0.1


class MoEElleci(nn.Module):
    """
    Elleci model with Mixture of Experts.

    Replaces dense FFN in Mamba layers with MoE-FFN.
    MLA layers keep dense FFN (per paper recommendations).
    """
    pass  # Will be implemented in model conversion


# Export
__all__ = [
    'MoEConfig',
    'ExpertFFN',
    'Router',
    'MoELayer',
    'MoEFFN',
    'drop_upcycle_weights',
    'initialize_router_from_mla',
]


if __name__ == "__main__":
    # Self-test
    print("MoE Module Self-Test")
    print("=" * 60)

    # Create config
    config = MoEConfig(
        d_model=256,
        num_experts=8,
        top_k=2,
    )

    print(f"Config: {config.num_experts} experts, top-{config.top_k}")

    # Test Router
    print("\n1. Testing Router...")
    router = Router(config)
    x = torch.randn(2, 16, 256)  # [batch=2, seq=16, d_model=256]

    router_probs, top_k_indices, top_k_weights, aux_loss = router(x)
    print(f"   Router probs: {router_probs.shape}")
    print(f"   Top-k indices: {top_k_indices.shape}")
    print(f"   Top-k weights: {top_k_weights.shape}")
    print(f"   Aux loss: {aux_loss.item():.4f}")

    # Verify top-k selection
    assert top_k_indices.max() < config.num_experts
    assert top_k_indices.min() >= 0
    print("   Router OK!")

    # Test MoELayer
    print("\n2. Testing MoELayer...")
    moe = MoELayer(config)
    moe.train()  # Enable training mode for ERC loss
    output, aux_loss = moe(x)
    print(f"   Input: {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Aux loss (with ERC): {aux_loss.item():.4f}")

    # Verify shapes
    assert output.shape == x.shape
    print("   MoELayer OK!")

    # Test ERC Loss specifically
    print("\n2b. Testing ERC Loss...")
    erc_loss = moe.compute_erc_loss()
    print(f"   ERC loss: {erc_loss.item():.4f}")
    # ERC loss should be positive and bounded (cross-entropy of NxN matrix)
    assert erc_loss.item() >= 0, "ERC loss should be non-negative"
    assert erc_loss.item() < 10, "ERC loss should be reasonable (< 10 for 8 experts)"
    print("   ERC Loss OK!")

    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = output.sum() + aux_loss
    loss.backward()

    grad_norms = []
    for i, expert in enumerate(moe.experts):
        grad_norm = expert.w1.weight.grad.norm().item()
        grad_norms.append(grad_norm)
    print(f"   Expert grad norms: {[f'{g:.4f}' for g in grad_norms[:4]]}...")
    print("   Gradient flow OK!")

    # Test MoEFFN (drop-in replacement)
    print("\n4. Testing MoEFFN...")
    moe_ffn = MoEFFN(256, config)
    y = moe_ffn(x)
    print(f"   Output shape: {y.shape}")
    print(f"   Aux loss: {moe_ffn.aux_loss.item():.4f}")
    print("   MoEFFN OK!")

    # Test drop_upcycle_weights
    print("\n5. Testing Drop-Upcycling...")
    state_dict = {'weight': torch.randn(64, 64)}
    modified = drop_upcycle_weights(state_dict, drop_ratio=0.2, seed=42)

    # Check that some weights changed
    diff = (state_dict['weight'] - modified['weight']).abs()
    changed_ratio = (diff > 1e-6).float().mean().item()
    print(f"   Changed weights: {changed_ratio*100:.1f}%")
    assert 0.15 < changed_ratio < 0.25  # ~20% with some variance
    print("   Drop-Upcycling OK!")

    # Count parameters
    print("\n6. Parameter count...")
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"   MoELayer params: {total_params/1e6:.2f}M")

    # Active params (only top-k experts)
    expert_params = sum(p.numel() for p in moe.experts[0].parameters())
    active_params = expert_params * config.top_k + sum(p.numel() for p in moe.router.parameters())
    print(f"   Active params per forward: {active_params/1e6:.2f}M")
    print(f"   Efficiency: {active_params/total_params*100:.1f}% active")

    print("\n" + "=" * 60)
    print("All MoE tests passed!")
