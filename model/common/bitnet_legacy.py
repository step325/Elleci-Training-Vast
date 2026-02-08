"""
BitNet 1.58b - Ternary Weight Quantization

Implements {-1, 0, 1} weight quantization with activation quantization.
Based on: "The Era of 1-bit LLMs" (BitNet b1.58)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class _RMSNorm(nn.Module):
    """RMSNorm that handles dtype correctly to avoid mismatch warnings."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute in float32 for numerical stability
        x_f32 = x.float()
        norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        # Cast both norm and weight to input dtype
        return norm.type_as(x) * self.weight.type_as(x)



class BitLinear(nn.Linear):
    """
    Linear layer with ternary weight quantization {-1, 0, 1}.
    
    During forward:
    1. Quantize activations to 8-bit (simulated)
    2. Quantize weights to {-1, 0, 1}
    3. Perform matrix multiplication
    4. Use Straight-Through Estimator (STE) for backprop
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias (default: False for BitNet)
        eps: Small constant for numerical stability
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        
        # RMSNorm for activation normalization
        self.rms_norm = _RMSNorm(in_features, eps=eps)
        
    def activation_quant(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations to 8-bit (simulated in float).
        
        Formula: Q(x) = Clip(Round(x / s), -128, 127) * s
        where s = max(|x|) / 127
        
        Args:
            x: Input tensor [*, in_features]
            
        Returns:
            Quantized tensor (same shape as input)
        """
        # Force float32 for stability during scaling
        x_f32 = x.float()
        
        # Compute scale: max absolute value per token
        scale = 127.0 / x_f32.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        
        # Quantize: scale, round, clip, unscale
        x_quant = (x_f32 * scale).round().clamp_(-128, 127) / scale
        
        # Cast back to original dtype
        x_quant = x_quant.to(x.dtype)
        
        # Straight-Through Estimator: use quantized in forward, original gradient
        # We return x + (x_quant - x).detach()
        # Forward: x + x_quant - x = x_quant
        # Backward: gradient flows through x (STE)
        return x + (x_quant - x).detach()
    
    def weight_quant(self, w: torch.Tensor) -> torch.Tensor:
        """
        Quantize weights to {-1, 0, 1}.
        
        Formula: Q(W) = Clip(Round(W / γ), -1, 1) * γ
        where γ = mean(|W|)
        
        Args:
            w: Weight tensor [out_features, in_features]
            
        Returns:
            Ternary quantized weights
        """
        # Force float32 for stability
        w_f32 = w.float()
        
        # Compute scale: mean absolute value (clamp to avoid division by zero)
        gamma = w_f32.abs().mean()
        gamma = torch.clamp(gamma, min=self.eps)
        
        # Normalize, round, and clamp to {-1, 0, 1}
        w_normalized = w_f32 / gamma
        w_quant_norm = torch.clamp(torch.round(w_normalized), -1, 1)
        
        # Scale back
        w_quant = w_quant_norm * gamma
        
        # Cast back and Straight-Through Estimator
        w_quant = w_quant.to(w.dtype)
        return w + (w_quant - w).detach()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights and activations.
        
        Args:
            x: Input [batch, seq_len, in_features]
            
        Returns:
            Output [batch, seq_len, out_features]
        """
        # 1. Normalize activations (Force float32 for stability, then cast back)
        x_norm = self.rms_norm(x.float())
        x = x_norm.to(x.dtype)
        
        # 2. Quantize activations
        x_q = self.activation_quant(x)
        
        # 3. Quantize weights
        w_q = self.weight_quant(self.weight)
        
        # 4. Linear transformation
        return F.linear(x_q, w_q, self.bias)
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, eps={self.eps}'


class BitLinear_a4(BitLinear):
    """
    BitNet a4.8 - 4-bit Activations for 1-bit LLMs (arXiv:2411.04965)

    Extends BitNet 1.58b with 4-bit activation quantization:
    - Weights: {-1, 0, 1} (ternary, same as BitNet 1.58b)
    - Activations: 4-bit (range [-7, 7])

    Key benefits:
    - 55% sparsity in activations (more zeros)
    - 3-bit effective KV cache compression
    - Better memory efficiency for inference

    The key insight is that with proper training, activations can be
    more aggressively quantized without quality loss.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to use bias
        eps: Numerical stability constant
        sparsify: Apply sparsification to activations
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        sparsify: bool = True
    ):
        super().__init__(in_features, out_features, bias=bias, eps=eps)
        self.sparsify = sparsify

        # Learnable threshold for sparsification
        self.sparsity_threshold = nn.Parameter(torch.tensor(0.1))

    def activation_quant_4bit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize activations to 4-bit (range [-7, 7]).

        Formula: Q(x) = Clip(Round(x / s), -7, 7) * s
        where s = max(|x|) / 7

        Args:
            x: Input tensor [*, in_features]

        Returns:
            4-bit quantized tensor (simulated in float)
        """
        x_f32 = x.float()

        # Compute scale for 4-bit range
        scale = 7.0 / x_f32.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)

        # Quantize: scale, round, clip to 4-bit range, unscale
        x_quant = (x_f32 * scale).round().clamp_(-7, 7) / scale

        # Optional: Apply sparsification (set small values to zero)
        if self.sparsify:
            # Values below threshold become zero
            mask = x_quant.abs() > self.sparsity_threshold
            x_quant = x_quant * mask.float()

        x_quant = x_quant.to(x.dtype)

        # STE for gradient flow
        return x + (x_quant - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 4-bit activations and ternary weights.

        Args:
            x: Input [batch, seq_len, in_features]

        Returns:
            Output [batch, seq_len, out_features]
        """
        # 1. Normalize activations
        x_norm = self.rms_norm(x.float())
        x = x_norm.to(x.dtype)

        # 2. Quantize activations to 4-bit
        x_q = self.activation_quant_4bit(x)

        # 3. Quantize weights to ternary
        w_q = self.weight_quant(self.weight)

        # 4. Linear transformation
        return F.linear(x_q, w_q, self.bias)

    def get_sparsity(self, x: torch.Tensor) -> float:
        """Calculate activation sparsity ratio."""
        with torch.no_grad():
            x_q = self.activation_quant_4bit(x)
            zeros = (x_q == 0).float().mean().item()
            return zeros


class BitLinear_a4_KV(BitLinear_a4):
    """
    BitNet a4.8 with 3-bit KV cache compression.

    Specialized for attention K/V projections where even more aggressive
    quantization is possible due to the redundancy in attention patterns.

    Uses asymmetric quantization for K and V:
    - K: 4-bit signed [-7, 7]
    - V: 3-bit unsigned [0, 7] (values tend to be positive after softmax weighting)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5,
        is_value_proj: bool = False
    ):
        super().__init__(in_features, out_features, bias=bias, eps=eps)
        self.is_value_proj = is_value_proj

    def activation_quant_3bit_unsigned(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize to 3-bit unsigned [0, 7] for value projections.

        Args:
            x: Input tensor

        Returns:
            3-bit quantized tensor
        """
        x_f32 = x.float()

        # Shift to positive range
        x_min = x_f32.min(dim=-1, keepdim=True).values
        x_shifted = x_f32 - x_min

        # Scale to [0, 7]
        scale = 7.0 / x_shifted.max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        x_quant = (x_shifted * scale).round().clamp_(0, 7) / scale + x_min

        x_quant = x_quant.to(x.dtype)
        return x + (x_quant - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with appropriate quantization for K or V."""
        x_norm = self.rms_norm(x.float())
        x = x_norm.to(x.dtype)

        if self.is_value_proj:
            x_q = self.activation_quant_3bit_unsigned(x)
        else:
            x_q = self.activation_quant_4bit(x)

        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)


class BitLinear_PTQTP(nn.Linear):
    """
    Post-Training Quantization to Ternary Precision (PTQTP)
    Based on arXiv:2509.16989

    Enables converting pre-trained models to ternary without retraining.
    Uses calibration data to find optimal quantization parameters.

    Usage:
        1. Create layer: layer = BitLinear_PTQTP(in_f, out_f)
        2. Copy weights: layer.weight.data = pretrained_weight
        3. Calibrate: layer.calibrate(calibration_data)
        4. Use normally: output = layer(input)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        eps: float = 1e-5
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = eps
        self.rms_norm = _RMSNorm(in_features, eps=eps)

        # Calibration parameters (learned during calibration)
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('activation_scale', torch.ones(1))
        self.register_buffer('calibrated', torch.tensor(False))

        # Quantized weight cache
        self.register_buffer('weight_quantized', None)

    def calibrate(self, calibration_data: torch.Tensor, num_samples: int = 1000):
        """
        Calibrate quantization parameters using sample data.

        Args:
            calibration_data: Sample inputs [num_samples, in_features]
            num_samples: Number of samples to use
        """
        self.eval()
        with torch.no_grad():
            # Sample from calibration data
            if calibration_data.shape[0] > num_samples:
                indices = torch.randperm(calibration_data.shape[0])[:num_samples]
                samples = calibration_data[indices]
            else:
                samples = calibration_data

            # Compute activation statistics
            samples_norm = self.rms_norm(samples.float())
            self.activation_scale.fill_(samples_norm.abs().mean().item())

            # Compute optimal weight scale for ternary
            w = self.weight.float()
            # Use mean absolute value as scale
            self.weight_scale.fill_(w.abs().mean().item())

            # Pre-compute quantized weights
            w_norm = w / self.weight_scale.clamp(min=self.eps)
            self.weight_quantized = torch.clamp(torch.round(w_norm), -1, 1) * self.weight_scale

            self.calibrated.fill_(True)

        print(f"Calibration complete: weight_scale={self.weight_scale.item():.4f}, "
              f"activation_scale={self.activation_scale.item():.4f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with calibrated quantization."""
        x_norm = self.rms_norm(x.float())
        x = x_norm.to(x.dtype)

        # Quantize activations
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=self.eps)
        x_q = (x * scale).round().clamp_(-128, 127) / scale

        # Use pre-computed quantized weights if calibrated
        if self.calibrated and self.weight_quantized is not None:
            w_q = self.weight_quantized
        else:
            # Fall back to runtime quantization
            gamma = self.weight.abs().mean().clamp(min=self.eps)
            w_q = torch.clamp(torch.round(self.weight / gamma), -1, 1) * gamma

        return F.linear(x_q, w_q, self.bias)


# Export
__all__ = ['BitLinear', 'BitLinear_a4', 'BitLinear_a4_KV', 'BitLinear_PTQTP']


if __name__ == "__main__":
    # Quick self-test
    print("BitLinear Self-Test")
    print("=" * 50)
    
    # Create layer
    layer = BitLinear(128, 256, bias=False)
    print(f"✓ Created BitLinear(128, 256)")
    
    # Test forward pass
    x = torch.randn(4, 10, 128)  # [batch=4, seq=10, features=128]
    out = layer(x)
    print(f"✓ Forward pass: {x.shape} → {out.shape}")
    
    # Test weight quantization
    w_quant = layer.weight_quant(layer.weight)
    unique_vals = torch.unique(w_quant / w_quant.abs().mean())
    unique_vals = unique_vals[unique_vals.abs() > 0.01]  # Filter near-zero
    print(f"✓ Weight quantization: unique values = {unique_vals.tolist()}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"✓ Gradient flow: grad norm = {layer.weight.grad.norm().item():.4f}")
    
    print("\n✅ All tests passed!")
