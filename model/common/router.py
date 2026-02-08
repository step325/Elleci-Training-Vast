"""
Adaptive Router - Fast vs Slow Path Selection

Decides whether to use:
- Fast path: 2 shallow layers (System 1)
- Slow path: Recurrent thinking loop (System 2)

Based on input complexity/confidence.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveRouter(nn.Module):
    """
    Router network for adaptive computation.
    
    Takes hidden state and outputs routing probabilities.
    Uses simple MLP classifier.
    
    Args:
        config: RouterConfig with d_model, router_hidden, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.router_hidden = config.router_hidden
        self.temperature = config.temperature
        
        # Gating network (MLP)
        self.gate = nn.Sequential(
            nn.Linear(config.d_model, config.router_hidden),
            nn.GELU(),
            nn.Linear(config.router_hidden, 2),  # Fast vs Slow
        )
        
    def forward(self, x, hard=False):
        """
        Compute routing probabilities (CAUSAL - per token).
        
        Args:
            x: Input [batch, seq_len, d_model] - MUST keep sequence dimension!
            hard: If True, return one-hot (for inference)
            
        Returns:
            probs: [batch, seq, 2] - [P(fast), P(slow)] PER TOKEN (causal!)
        """
        # CRITICAL: Do NOT pool over sequence!
        # Old code: x_pooled = x.mean(dim=1)  ← BREAKS CAUSALITY!
        # New code: Process each token independently
        
        # Compute logits per-token
        logits = self.gate(x)  # [batch, seq, 2] - CAUSAL!
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Softmax probabilities (per token)
        probs = F.softmax(logits, dim=-1)  # [batch, seq, 2]
        
        if hard:
            # Gumbel-Softmax for differentiable hard routing
            probs_hard = F.gumbel_softmax(logits, tau=1.0, hard=True)
            return probs_hard, probs_hard.argmax(dim=-1)
        
        return probs
    
    def route_decision(self, x, threshold=0.5):
        """
        Make routing decision based on threshold.
        
        Args:
            x: Input
            threshold: Probability threshold for slow path
            
        Returns:
            use_slow: Boolean whether to use slow path
        """
        probs = self.forward(x)
        use_slow = probs[:, 1] > threshold
        return use_slow


# Export
__all__ = ['AdaptiveRouter']


if __name__ == "__main__":
    # Self-test
    print("Router Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        d_model: int = 768
        router_hidden: int = 256
        temperature: float = 1.0
        n_experts: int = 2
    
    config = TestConfig()
    router = AdaptiveRouter(config)
    
    print(f"✓ Created Router (d_model={config.d_model})")
    
    # Test with 2D input
    x = torch.randn(4, 768)  # [batch, d_model]
    probs = router(x)
    print(f"✓ 2D input: {x.shape} → probs={probs.shape}")
    print(f"  Sample probs: Fast={probs[0,0]:.3f}, Slow={probs[0,1]:.3f}")
    
    # Test with 3D input
    x_seq = torch.randn(4, 16, 768)  # [batch, seq, d_model]
    probs_seq = router(x_seq)
    print(f"✓ 3D input: {x_seq.shape} → probs={probs_seq.shape}")
    
    # Test hard routing
    probs_hard, route = router(x, hard=True)
    print(f"✓ Hard routing: {route} (0=fast, 1=slow)")
    
    # Test gradient flow
    loss = probs.sum()
    loss.backward()
    print(f"✓ Gradient flow: gate[0] grad norm = {router.gate[0].weight.grad.norm().item():.4f}")
    
    # Test routing decision
    use_slow = router.route_decision(x, threshold=0.5)
    print(f"✓ Route decision: slow_path={use_slow.sum().item()}/{len(use_slow)}")
    
    print("\n✅ All tests passed!")
