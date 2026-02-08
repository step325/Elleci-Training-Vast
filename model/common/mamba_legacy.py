"""
Mamba - Selective State Space Model

Simplified implementation of Mamba for sequence modeling.
Key features:
- Linear complexity O(N) instead of quadratic O(N^2) attention
- Selective scan mechanism
- State space formulation

Based on: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class MambaBlock(nn.Module):
    """
    Mamba block with selective state space mechanism.
    
    Architecture:
    1. Expand input via linear projection
    2. Apply depthwise convolution (local context)
    3. Selective scan (SSM) for long-range dependencies
    4. Gate and project back
    
    Args:
        config: MambaConfig with d_model, d_state, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand
        self.d_inner = int(self.expand * self.d_model)
        
        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,  # Depthwise
            padding=self.d_conv - 1,
        )
        
        # SSM parameters (simplified)
        # In full Mamba, these are input-dependent (selective)
        # For simplicity, we make them learnable but fixed
        self.x_proj = nn.Linear(self.d_inner, config.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, self.d_inner, bias=True)
        
        # State space parameters A and D
        # A: [d_inner, d_state] - state transition matrix
        # D: [d_inner] - skip connection
        A = torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x):
        """
        Forward pass through Mamba block.
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seqlen, dim = x.shape
        
        # 1. Expand input
        x_expanded = self.in_proj(x)  # [batch, seq, d_inner*2]
        x_expanded, z = x_expanded.chunk(2, dim=-1)  # Split for gating
        
        # 2. Depthwise convolution (local context)
        x_conv = rearrange(x_expanded, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seqlen]  # Trim padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        
        x_conv = self.act(x_conv)
        
        # 3. SSM (selective scan) - simplified version
        # Full Mamba has complex selective scan, we use simplified state space
        y = self.selective_scan(x_conv)
        
        # 4. Gate with z
        y = y * self.act(z)
        
        # 5. Output projection
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, x):
        """
        Selective scan mechanism using parallel scan (Blelloch algorithm).
        
        Time complexity: O(log N) instead of O(N).
        
        Args:
            x: [batch, seq_len, d_inner]
            
        Returns:
            y: [batch, seq_len, d_inner]
        """
        from .pscan import pscan
        
        batch, seqlen, d_inner = x.shape
        
        # Get SSM parameters (input-dependent)
        x_dbl = self.x_proj(x)  # [batch, seq, dt_rank + d_state*2]
        
        dt_rank = x_dbl.shape[-1] - self.d_state * 2
        delta = x_dbl[..., :dt_rank]  # Time step
        B = x_dbl[..., dt_rank:dt_rank+self.d_state]  # Input matrix [batch, seq, d_state]
        C = x_dbl[..., dt_rank+self.d_state:]  # Output matrix [batch, seq, d_state]
        
        # Process delta
        delta = self.dt_proj(delta)  # [batch, seq, d_inner]
        delta = F.softplus(delta)  # Ensure positive
        
        # Get A (state transition) - [d_inner, d_state]
        A = -torch.exp(self.A_log.float())
        
        # Discretize A: deltaA = exp(delta * A)
        # delta: [batch, seq, d_inner] -> [batch, seq, d_inner, 1]
        # A: [d_inner, d_state] -> [1, 1, d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [batch, seq, d_inner, d_state]
        
        # Compute deltaB * x
        # B: [batch, seq, d_state] -> [batch, seq, 1, d_state]
        # x: [batch, seq, d_inner] -> [batch, seq, d_inner, 1]
        # delta: [batch, seq, d_inner] -> [batch, seq, d_inner, 1]
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # [batch, seq, d_inner, d_state]
        
        # Parallel scan: H[t] = A[t] * H[t-1] + X[t]
        # pscan expects (B, L, D, N) format
        # deltaA: [batch, seq, d_inner, d_state] -> already correct
        # deltaB_x: [batch, seq, d_inner, d_state] -> already correct
        H = pscan(deltaA, deltaB_x)  # [batch, seq, d_inner, d_state]
        
        # Output: y = sum(H * C) over state dimension
        # C: [batch, seq, d_state] -> [batch, seq, 1, d_state]
        y = (H * C.unsqueeze(2)).sum(dim=-1)  # [batch, seq, d_inner]
        
        # Add skip connection (D parameter)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


# Export
__all__ = ['MambaBlock']


if __name__ == "__main__":
    # Self-test
    print("Mamba Block Self-Test")
    print("=" * 50)
    
    from dataclasses import dataclass
    
    @dataclass
    class TestConfig:
        d_model: int = 768
        d_state: int = 16
        d_conv: int = 4
        expand: int = 2
        dt_rank: int = 48  # d_model // 16
    
    config = TestConfig()
    mamba = MambaBlock(config)
    
    print(f"✓ Created Mamba (d_model={config.d_model}, d_state={config.d_state})")
    
    # Test forward
    x = torch.randn(2, 32, 768)  # [batch=2, seq=32, d_model=768]
    print(f"Input shape: {x.shape}")
    
    out = mamba(x)
    print(f"✓ Forward pass: {x.shape} → {out.shape}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"✓ Gradient flow: in_proj grad norm = {mamba.in_proj.weight.grad.norm().item():.4f}")
    
    # Check complexity
    params = sum(p.numel() for p in mamba.parameters())
    print(f"✓ Parameters: {params/1e6:.2f}M")
    
    print("\n✅ All tests passed!")
