
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from .triton_kernels import unpack_2bit_triton


class BitNetFunction(Function):
    @staticmethod
    def forward(ctx, input, weight_packed, scale, output_shape_weight, layer_ref=None):
        """
        Forward pass with support for gradient extraction.
        Args:
            scale: [Out, 1] BF16 tensor.
        """
        # Unpack on the fly
        weight_bf16 = unpack_2bit_triton(weight_packed, output_shape_weight)
        
        # Matmul: Input @ Weight.T
        output_unscaled = torch.matmul(input, weight_bf16.t())
        
        ctx.save_for_backward(input, weight_packed, scale, output_unscaled)
        ctx.output_shape_weight = output_shape_weight
        ctx.layer_ref = layer_ref # Store reference to layer to save grad
        
        # Apply scale inside the function to capture gradient
        output = output_unscaled * scale.squeeze(-1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_packed, scale, output_unscaled = ctx.saved_tensors
        output_shape_weight = ctx.output_shape_weight
        
        # Re-unpack weights
        weight_bf16 = unpack_2bit_triton(weight_packed, output_shape_weight)
        
        # We reuse output_unscaled from forward to save compute and ensure correct grad calculation
        
        # 1. Scale Gradient
        # dL/ds = sum(dL/dY * Y_unscaled)
        # grad_output: [..., Out], output_unscaled: [..., Out]
        # Sum over all dims except last
        sum_dims = list(range(len(grad_output.shape) - 1))
        # Reduce in Float32 to avoid underflow
        grad_scale = (grad_output.float() * output_unscaled.float()).sum(dim=sum_dims, keepdim=True).to(scale.dtype)
        # Reshape to [Out, 1] to match scale param
        grad_scale = grad_scale.view_as(scale)

        # 2. Input Gradient
        # dL/dX = dL/dY * W_scaled = grad_output * (W * s) = (grad_output * s) @ W
        grad_input = None
        
        # Pre-scale grad_output
        grad_output_scaled = grad_output * scale.squeeze(-1)
        
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output_scaled, weight_bf16)
            
        # 3. Weight Gradient (Optimization Step)
        # dL/dW = (grad_output * s)^T @ Input
        grad_weight = None
        
        if ctx.layer_ref is not None:
            # Fuse batch dimensions
            g_out_flat = grad_output_scaled.reshape(-1, grad_output_scaled.shape[-1])
            input_flat = input.reshape(-1, input.shape[-1])
            
            grad_weight = torch.matmul(g_out_flat.t(), input_flat)
            
            # Store in the layer reference
            if ctx.layer_ref.weight_grad_shadow is None:
                ctx.layer_ref.weight_grad_shadow = grad_weight
            else:
                ctx.layer_ref.weight_grad_shadow += grad_weight
            
        # Return gradients matching forward signature:
        # input, weight_packed, scale, output_shape, layer_ref
        return grad_input, None, grad_scale, None, None

class BitNetLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        assert in_features % 4 == 0, "in_features must be divisible by 4 for 2-bit packing"
        
        # 1. Packed Weights (Int8, GPU)
        # Shape: [Out, In // 4]
        # requires_grad=False because we don't update this via Autograd
        self.register_parameter('weight_packed', 
            nn.Parameter(
                torch.zeros(out_features, in_features // 4, dtype=torch.int8),
                requires_grad=False
            )
        )
        
        # 2. Scale (BF16, GPU)
        # requires_grad=True -> Updated by Standard Optimizer (AdamW)
        self.scale = nn.Parameter(torch.ones(out_features, 1, dtype=torch.bfloat16))
        
        # 3. Bias (Optional, BF16, GPU)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter('bias', None)
            
        # 4. Master Weights (BF16, CPU)
        # Managed manually by Custom Optimizer
        # Not a Property/Buffer to avoid moving to GPU automatically
        self.master_weight = None
        self.weight_grad_shadow = None

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """Save master_weight to state_dict"""
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.master_weight is not None:
            destination[prefix + 'master_weight'] = self.master_weight

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Load master_weight from state_dict"""
        # Load standard params
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
        # Load master_weight
        key = prefix + 'master_weight'
        if key in state_dict:
            self.master_weight = state_dict[key]
        elif strict:
            missing_keys.append(key)

    def quantize_and_pack(self, weight_bf16=None):
        """
        Utility to quantize and pack BF16 weights into int8.
        If weight_bf16 is None, uses self.master_weight.
        Updates self.weight_packed on GPU.
        """
        if weight_bf16 is None:
            if self.master_weight is None:
                raise ValueError("Master weight not initialized")
            weight_bf16 = self.master_weight
            
        # Quantize to {-1, 0, 1}
        # Calculation on same device as weight_bf16 (likely CPU for master weights)
        scale_val = self.scale.data.to(weight_bf16.device) + 1e-6
        w_scaled = weight_bf16 / scale_val
        w_quant = torch.round(w_scaled).clamp(-1, 1).to(torch.int8)
        
        # Map values to 2-bit codes:
        # -1 -> 2 (binary 10)
        # 0  -> 0 (binary 00)
        # 1  -> 1 (binary 01)
        
        # Vectorized map:
        # replace -1 with 2
        codes = w_quant.clone()
        codes[codes == -1] = 2
        
        # Pack
        # Reshape to [out, in//4, 4]
        # Ensure divisible by 4 (asserted in init)
        codes_reshaped = codes.view(self.out_features, -1, 4)
        
        # Bitwise shift
        packed = (codes_reshaped[..., 0] << 0) | \
                 (codes_reshaped[..., 1] << 2) | \
                 (codes_reshaped[..., 2] << 4) | \
                 (codes_reshaped[..., 3] << 6)
                 
        # Move to GPU and assign
        self.weight_packed.data.copy_(packed.to(device=self.weight_packed.device))

    def reset_parameters(self):
        # 1. Kaiming Normal Initialization for Master Weights
        # Using CPU to save VRAM from start
        self.master_weight = torch.empty(self.out_features, self.in_features, dtype=torch.bfloat16, device='cpu')
        nn.init.kaiming_normal_(self.master_weight, mode='fan_in', nonlinearity='linear')
        
        # 2. Initial Scale Calculation
        # BitNet Rule: scale = 1 / mean(abs(W))
        # This roughly ensures that quantized_W * scale has similar magnitude to original W
        # We compute this PER CHANNEL (dim=1)
        
        # avg_abs: [Out, 1]
        avg_abs = self.master_weight.abs().mean(dim=1, keepdim=True)
        
        # Avoid division by zero
        # Avoid division by zero
        self.scale.data = (avg_abs + 1e-6).to(dtype=self.scale.dtype, device=self.scale.device)
        
        # 3. Pack (CPU -> GPU)
        self.quantize_and_pack()

    def forward(self, input):
        # Ensure BF16
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)
            
        # Call Custom Function
        # Scale is applied INSIDE the function now
        out = BitNetFunction.apply(input, self.weight_packed, self.scale, (self.out_features, self.in_features), self)
        
        if self.bias is not None:
            out = out + self.bias
            
        return out
