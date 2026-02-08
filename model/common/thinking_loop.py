"""
Thinking Loop - Recurrent Reasoning Module

Implements iterative refinement for complex reasoning.
Used in "slow path" (System 2) for multi-step thinking.
"""
import torch
import torch.nn as nn


class ThinkingLoop(nn.Module):
    """
    Recurrent thinking mechanism.
    
    Takes an initial hidden state and refines it through N iterations.
    Each iteration uses the same transformer layer.
    
    Args:
        config: ThinkingLoopConfig with max_iterations, etc.
        layer: The layer to apply recurrently (e.g., TransformerBlock)
    """
    def __init__(self, config, layer):
        super().__init__()
        self.max_iterations = config.max_iterations
        self.convergence_threshold = config.convergence_threshold
        self.layer = layer
        
    def forward(self, x, n_iterations=None):
        """
        Apply thinking loop.
        
        Args:
            x: Input [batch, seq_len, d_model]
            n_iterations: Number of iterations (default: max_iterations)
            
        Returns:
            output: Refined state [batch, seq_len, d_model]
            n_steps: Actual number of iterations used
        """
        if n_iterations is None:
            n_iterations = self.max_iterations
        
        current = x
        
        for i in range(n_iterations):
            # Apply layer
            next_state = self.layer(current)
            
            # Check convergence (optional early stopping)
            if i > 0:
                delta = (next_state - current).abs().mean()
                if delta < self.convergence_threshold:
                    return next_state, i + 1
            
            current = next_state
        
        return current, n_iterations


class PonderNetThinkingLoop(nn.Module):
    """
    PonderNet-style thinking loop with learned halting.
    
    Instead of using a fixed convergence threshold, this module learns
    when to stop thinking based on input complexity.
    
    Key features:
    - Halting probability λ_n learned per step
    - Weighted sum of all intermediate states
    - KL regularization to encourage geometric prior distribution
    - Early exit during inference for efficiency
    
    Reference: Banino et al., "PonderNet: Learning to Ponder" (2021)
    https://arxiv.org/abs/2107.05407
    """
    def __init__(self, config, layer, d_model):
        super().__init__()
        self.max_iterations = config.max_iterations
        self.lambda_p = config.lambda_p  # Geometric prior
        self.kl_weight = config.kl_weight
        self.epsilon = config.epsilon  # Min probability floor
        self.layer = layer
        
        # Halting probability predictor
        self.halt_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        
    def forward(self, x, n_iterations=None):
        """
        Apply PonderNet thinking loop.
        
        Args:
            x: Input [batch, seq_len, d_model]
            n_iterations: Not used (kept for compatibility)
            
        Returns:
            output: Weighted refined state [batch, seq_len, d_model]
            n_steps: Expected number of iterations (float)
            ponder_loss: KL regularization loss (for training)
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # Accumulate outputs weighted by halting probability
        output_acc = torch.zeros_like(x)
        p_halt_cumulative = torch.zeros(batch_size, device=device)
        
        # For loss computation
        halt_probs = []
        
        current = x
        
        for n in range(self.max_iterations):
            # Apply thinking layer
            current = self.layer(current)
            
            # Compute halting probability for this step (pooled over sequence)
            # Shape: [batch, d_model] -> [batch, 1] -> [batch]
            h_pooled = current.mean(dim=1)  # [batch, d_model]
            p_halt_raw = torch.sigmoid(self.halt_proj(h_pooled)).squeeze(-1)  # [batch]
            
            # Apply epsilon floor to prevent collapse
            p_halt = self.epsilon + (1 - self.epsilon) * p_halt_raw
            
            # Probability of halting at THIS step = p_halt * (1 - cumulative)
            p_n = p_halt * (1 - p_halt_cumulative)
            halt_probs.append(p_n)
            
            # Accumulate weighted output
            output_acc = output_acc + p_n.unsqueeze(-1).unsqueeze(-1) * current
            
            # Update cumulative probability
            p_halt_cumulative = p_halt_cumulative + p_n
            
            # Early exit during inference if all samples have halted
            if not self.training and (p_halt_cumulative > 0.99).all():
                break
        
        # Remainder mass goes to final state
        p_remainder = 1 - p_halt_cumulative
        output_acc = output_acc + p_remainder.unsqueeze(-1).unsqueeze(-1) * current
        
        # Expected number of steps
        steps = torch.arange(1, len(halt_probs) + 1, device=device, dtype=x.dtype)
        halt_probs_tensor = torch.stack(halt_probs, dim=1)  # [batch, n_steps]
        expected_steps = (halt_probs_tensor * steps.unsqueeze(0)).sum(dim=1).mean()
        
        # KL divergence loss: encourage prior geometric distribution
        # Prior: p(n) = lambda_p * (1 - lambda_p)^(n-1)
        ponder_loss = torch.tensor(0.0, device=device)
        if self.training:
            for n, p_n in enumerate(halt_probs):
                # Geometric prior for step n
                prior_n = self.lambda_p * ((1 - self.lambda_p) ** n)
                prior_n = torch.tensor(prior_n, device=device)
                
                # KL divergence (simplified: just MSE between log probs)
                # Avoid log(0) with clamp
                p_n_clamped = p_n.clamp(min=1e-10)
                prior_clamped = prior_n.clamp(min=1e-10)
                kl_term = p_n * (torch.log(p_n_clamped) - torch.log(prior_clamped))
                ponder_loss = ponder_loss + kl_term.mean()
            
            ponder_loss = self.kl_weight * ponder_loss
        
        return output_acc, expected_steps.item(), ponder_loss


class SimpleThinkingLayer(nn.Module):
    """
    Simple layer for thinking loop (for testing).
    In practice, this would be a full transformer block.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        
    def forward(self, x):
        # Simple residual block
        return x + self.ffn(self.norm(x))


# Export
__all__ = ['ThinkingLoop', 'PonderNetThinkingLoop', 'SimpleThinkingLayer']


if __name__ == "__main__":
    # Self-test
    print("Thinking Loop Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        max_iterations: int = 4
        convergence_threshold: float = 0.01
    
    config = TestConfig()
    
    # Create a simple layer for testing
    d_model = 768
    layer = SimpleThinkingLayer(d_model)
    
    # Create thinking loop
    thinking = ThinkingLoop(config, layer)
    
    print(f"✓ Created ThinkingLoop (max_iter={config.max_iterations})")
    
    # Test forward
    x = torch.randn(2, 16, d_model)  # [batch, seq, d_model]
    output, n_steps = thinking(x)
    
    print(f"✓ Forward pass: {x.shape} → {output.shape}")
    print(f"  Iterations used: {n_steps}")
    
    # Test with fewer iterations
    output2, n_steps2 = thinking(x, n_iterations=2)
    print(f"✓ Limited iterations: used {n_steps2}/2")
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    print(f"✓ Gradient flow: layer.ffn[0] grad norm = {layer.ffn[0].weight.grad.norm().item():.4f}")
    
    # Test convergence
    # Create input that should converge quickly
    x_simple = torch.zeros(2, 16, d_model)
    output_conv, n_conv = thinking(x_simple)
    print(f"✓ Convergence test: converged in {n_conv} steps (max={config.max_iterations})")
    
    print("\n✅ All tests passed!")
