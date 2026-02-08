"""
Int2Linear with Tensor Core optimized kernels.

Drop-in replacement for Int2Linear using the TC-optimized CUDA kernels.
Achieves ~16x speedup on matmul operations via WMMA Tensor Cores.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple, Dict, Any
import math
import sys
import os

from torch.utils.cpp_extension import load

# JIT load core INT2 kernels from consolidated directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(CURRENT_DIR, "../../kernels/int2")

# Auto-detect GPU architecture for kernel compilation
def _get_cuda_arch_flag():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f'-arch=sm_{major}{minor}'
    return '-arch=sm_80'  # fallback to A100

_CUDA_ARCH = _get_cuda_arch_flag()

try:
    int2_tc_ops = load(
        name=f"int2_tc_ops_consolidated_v4_{_CUDA_ARCH.replace('-arch=', '')}",
        sources=[
            os.path.join(KERNELS_DIR, "int2_tc_ops.cpp"),
            os.path.join(KERNELS_DIR, "int2_matmul_tc.cu"),
            os.path.join(KERNELS_DIR, "int2_backward_tc.cu"),
            os.path.join(KERNELS_DIR, "int2_hysteresis_tc.cu"),
            os.path.join(KERNELS_DIR, "int2_hysteresis_v2.cu"),
            os.path.join(KERNELS_DIR, "int2_activation_quant.cu"),
            os.path.join(KERNELS_DIR, "int2_unpack.cu"),
        ],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math', _CUDA_ARCH],
        verbose=False
    )
    HAS_TC_OPS = True
    HAS_INT2_OPS = True  # consolidated includes pack/unpack logic
    print(f"[INT2] Kernels compiled for {_CUDA_ARCH}")
except Exception as e:
    print(f"FAILED to JIT load INT2 kernels: {e}")
    HAS_TC_OPS = False
    HAS_INT2_OPS = False


class Int2LinearTCFunction(Function):
    """Custom autograd function using TC-optimized kernels."""

    @staticmethod
    def forward(ctx, X: torch.Tensor, W_packed: torch.Tensor, gamma: torch.Tensor,
                K: int, gamma_cached: float = None) -> torch.Tensor:
        """
        Forward pass: Y = X @ W.T * gamma (using Tensor Cores)
        """
        ctx.save_for_backward(X, W_packed, gamma)
        ctx.K = K
        ctx.gamma_cached = gamma_cached

        X_half = X.half() if X.dtype != torch.float16 else X
        # Use cached gamma (Python float) to avoid GPU→CPU sync
        gamma_val = gamma_cached if gamma_cached is not None else gamma.item()

        # Use TC-optimized matmul
        Y = int2_tc_ops.matmul(X_half.contiguous(), W_packed, gamma_val)

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: compute dX = dY @ W (using Tensor Cores)
        """
        X, W_packed, gamma = ctx.saved_tensors
        K = ctx.K

        dX = None
        if ctx.needs_input_grad[0]:
            dY_half = dY.half() if dY.dtype != torch.float16 else dY
            # Use cached gamma to avoid GPU→CPU sync
            gamma_val = ctx.gamma_cached if ctx.gamma_cached is not None else gamma.item()

            # Use TC-optimized backward
            dX = int2_tc_ops.backward_input(dY_half.contiguous(), W_packed, gamma_val, K)

        return dX, None, None, None, None


class Int2LinearTC(nn.Module):
    """
    Linear layer with INT2 packed weights using Tensor Core optimized kernels.

    This is a drop-in replacement for Int2Linear with ~16x faster matmul.

    Memory footprint per weight:
    - Weight: 0.25 byte (INT2, 4 weights per byte)
    - Hysteresis: 0.5 byte (INT4, 2 counters per byte)
    - Total: 0.75 byte/weight

    Performance:
    - Forward: ~16x faster than original (29 TFLOPS on RTX 4070)
    - Backward: ~17x faster than original (30 TFLOPS on RTX 4070)

    INT8 Activation Compression (Phase 2):
    - Saved activations are compressed to INT8 (50% memory reduction)
    - Automatic quantize/dequantize with minimal overhead (~0.6ms)
    """

    # Shared absmax buffer for async quantization (avoids cudaStreamSynchronize)
    _shared_absmax_buf: Optional[torch.Tensor] = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: int = 7,
        lr_scale: float = 5.0,
        decay_rate: float = 0.001,
        compress_activations: bool = True  # Phase 2: Enable INT8 compression
    ):
        super().__init__()

        if not HAS_TC_OPS:
            raise RuntimeError("int2_tc_ops not available. Build with: cd cuda && python setup.py install")

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.lr_scale = lr_scale
        self.decay_rate = decay_rate
        self.compress_activations = compress_activations

        # Packed weights: INT2, 4 weights per byte
        packed_K = (in_features + 3) // 4
        self.register_buffer('W_packed', torch.zeros(out_features, packed_K, dtype=torch.uint8))

        # Hysteresis state: INT4, 2 counters per byte
        packed_H = (in_features + 1) // 2
        self.register_buffer('H_packed', torch.zeros(out_features, packed_H, dtype=torch.uint8))

        # Scale factor (learnable)
        self.gamma = nn.Parameter(torch.ones(1))

        # Bias (optional)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Training step counter (buffer for checkpoint compat, Python int for hot path)
        self.register_buffer('_step', torch.tensor(0, dtype=torch.int64))
        self._step_py = 0  # Python int, zero GPU→CPU sync

        # Cache for hysteresis update (Phase 2: compressed format)
        self._saved_input = None  # Will be (int8_tensor, scale) if compress_activations=True
        self._saved_input_shape = None  # Original shape for reconstruction
        self._initialized = False

        # Cached gamma value (Python float) to avoid .item() sync on every forward/backward
        self._gamma_cached = None

    def init_from_float(self, weight: torch.Tensor, scale: Optional[float] = None):
        """Initialize from float32 weights."""
        assert weight.shape == (self.out_features, self.in_features)

        if scale is None:
            scale = weight.abs().mean().item()
            scale = max(scale, 1e-5)

        w_scaled = weight / scale
        w_ternary = w_scaled.round().clamp(-1, 1).to(torch.int8)

        # Always use CPU packing (fast enough for initialization)
        self._pack_int2_cpu(w_ternary)

        self.gamma.data.fill_(scale)
        self._gamma_cached = scale  # Cache initial gamma
        self.H_packed.zero_()
        self._initialized = True

    def _pack_int2_cpu(self, w_ternary: torch.Tensor):
        """CPU fallback for packing."""
        N, K = w_ternary.shape
        packed_K = (K + 3) // 4
        packed = torch.zeros(N, packed_K, dtype=torch.uint8, device=w_ternary.device)

        w_cpu = w_ternary.cpu().numpy()
        packed_cpu = packed.cpu().numpy()

        for n in range(N):
            for k in range(K):
                byte_idx = k // 4
                bit_offset = (k % 4) * 2
                val = int(w_cpu[n, k])
                encoded = val + 1
                packed_cpu[n, byte_idx] |= (encoded << bit_offset)

        self.W_packed.copy_(torch.from_numpy(packed_cpu).to(w_ternary.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using TC-optimized matmul."""
        if not self._initialized:
            w_init = torch.randn(
                self.out_features, self.in_features,
                device=x.device, dtype=torch.float32
            ) * (1.0 / math.sqrt(self.in_features))
            self.init_from_float(w_init)

        if self.training:
            # Phase 2: Optionally compress saved activations to INT8
            x_detached = x.detach()
            if self.compress_activations:
                # Convert to half for quantization kernel
                x_half = x_detached.half() if x_detached.dtype != torch.float16 else x_detached
                self._saved_input_shape = x_half.shape
                # Async quantize: zero GPU→CPU sync
                if Int2LinearTC._shared_absmax_buf is None or Int2LinearTC._shared_absmax_buf.device != x.device:
                    Int2LinearTC._shared_absmax_buf = torch.zeros(1, dtype=torch.float32, device=x.device)
                self._saved_input = int2_tc_ops.quantize_activation_async(
                    x_half.contiguous(), Int2LinearTC._shared_absmax_buf
                )
            else:
                self._saved_input = x_detached
                self._saved_input_shape = None

        # Ensure gamma cache is set (first forward or after load)
        if self._gamma_cached is None:
            self._gamma_cached = self.gamma.detach().item()

        out = Int2LinearTCFunction.apply(x, self.W_packed, self.gamma, self.in_features, self._gamma_cached)

        if self.bias is not None:
            out = out + self.bias

        return out

    def hysteresis_update(self, dY: torch.Tensor, lr: float):
        """
        Apply hysteresis update to weights.
        
        ========================================================================
        OPTIMIZATION v2 (2024-02-04): SEPARATED dW/Hysteresis
        ========================================================================
        
        OLD APPROACH (SLOW - O(M×N×K)):
        - Single fused kernel that computed gradient AND applied hysteresis
        - For each weight: grad = sum_m dY[m,n] * X[m,k]  <- O(M) loop!
        - Total complexity: O(M × N × K) = extremely slow
        
        NEW APPROACH (FAST - O(N×K)):
        - Phase 1: Compute dW = dY.T @ X using torch.mm (cuBLAS Tensor Cores)
        - Phase 2: Apply hysteresis using pre-computed dW (new v2 kernel)
        - Total complexity: O(N×K) for hysteresis + O(M×N×K) with Tensor Cores
        
        SPEEDUP: ~9x faster (from ~10ms to ~1ms per layer)
        
        TRADE-OFF:
        - Pro: Uses cuBLAS Tensor Cores for gradient computation
        - Con: Needs temporary dW buffer (~4MB per layer for 1024×1024)
        ========================================================================
        """
        if self._saved_input is None:
            raise RuntimeError("No saved input. Call forward() first in training mode.")

        self._step_py += 1

        # Phase 2: Decompress saved input if it was compressed
        if self.compress_activations and isinstance(self._saved_input, tuple):
            x_int8, scale = self._saved_input
            X = int2_tc_ops.dequantize_activation(x_int8, scale)
            # Restore original shape
            if self._saved_input_shape is not None:
                X = X.view(self._saved_input_shape)
        else:
            X = self._saved_input

        dY_2d = dY.reshape(-1, dY.size(-1)).half().contiguous()  # [M, N]
        X_2d = X.reshape(-1, X.size(-1)).half().contiguous()      # [M, K]
        
        M = dY_2d.size(0)
        N = dY_2d.size(1)  # out_features
        K = X_2d.size(1)   # in_features
        
        # ==================================================================
        # NEW: Compute dW using cuBLAS (Tensor Cores) - ~0.1ms
        # dW[N, K] = dY.T[N, M] @ X[M, K]
        # ==================================================================
        # Using torch.mm leverages cuBLAS which uses Tensor Cores on modern GPUs
        dW = torch.mm(dY_2d.t(), X_2d).float()  # [N, K] in FP32 for hysteresis
        
        # ==================================================================
        # NEW: Call v2 hysteresis kernel (lightweight - O(N×K))
        # ==================================================================
        # Check if v2 kernel is available, otherwise fall back to legacy
        if hasattr(int2_tc_ops, 'hysteresis_step_v2'):
            int2_tc_ops.hysteresis_step_v2(
                dW.contiguous(),
                self.W_packed, self.H_packed,
                lr, self.lr_scale, self.threshold, self.decay_rate,
                self._step_py
            )
        else:
            # Fall back to legacy fused kernel (slower)
            int2_tc_ops.hysteresis_step(
                dY_2d, X_2d,
                self.W_packed, self.H_packed,
                lr, self.lr_scale, self.threshold, self.decay_rate,
                self._step_py
            )

        self._saved_input = None
        self._saved_input_shape = None
    
    # ==========================================================================
    # LEGACY METHOD (kept for reference - DO NOT USE)
    # ==========================================================================
    # 
    # def hysteresis_update_LEGACY(self, dY: torch.Tensor, lr: float):
    #     """
    #     DEPRECATED: Old fused hysteresis update.
    #     
    #     This method used a single kernel that computed gradient AND hysteresis.
    #     It was SLOW because:
    #     - Each weight computed: grad = sum_m dY[m,n] * X[m,k]
    #     - This is O(M) per weight = O(M×N×K) total
    #     - Memory access was non-coalesced (each thread reads different rows)
    #     - Could not leverage Tensor Cores
    #     
    #     The new hysteresis_update() above is ~9x faster.
    #     """
    #     if self._saved_input is None:
    #         raise RuntimeError("No saved input. Call forward() first in training mode.")
    #
    #     self._step += 1
    #
    #     if self.compress_activations and isinstance(self._saved_input, tuple):
    #         x_int8, scale = self._saved_input
    #         X = int2_tc_ops.dequantize_activation(x_int8, scale)
    #         if self._saved_input_shape is not None:
    #             X = X.view(self._saved_input_shape)
    #     else:
    #         X = self._saved_input
    #
    #     dY_2d = dY.reshape(-1, dY.size(-1)).half().contiguous()
    #     X_2d = X.reshape(-1, X.size(-1)).half().contiguous()
    #
    #     # SLOW: Uses fused kernel that computes gradient inline - O(M×N×K)!
    #     int2_tc_ops.hysteresis_step(
    #         dY_2d, X_2d,
    #         self.W_packed, self.H_packed,
    #         lr, self.lr_scale, self.threshold, self.decay_rate,
    #         self._step.item()
    #     )
    #
    #     self._saved_input = None
    #     self._saved_input_shape = None
    # ==========================================================================



    def memory_footprint(self) -> Dict[str, Any]:
        """Calculate memory footprint statistics."""
        total_weights = self.out_features * self.in_features

        w_bytes = self.W_packed.numel()
        h_bytes = self.H_packed.numel()
        gamma_bytes = 4
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0

        total_bytes = w_bytes + h_bytes + gamma_bytes + bias_bytes
        bytes_per_weight = (w_bytes + h_bytes) / total_weights
        fp32_bytes = total_weights * 4
        compression = fp32_bytes / (w_bytes + h_bytes)

        return {
            'total_weights': total_weights,
            'total_bytes': total_bytes,
            'w_bytes': w_bytes,
            'h_bytes': h_bytes,
            'bytes_per_weight': bytes_per_weight,
            'compression_vs_fp32': compression,
            'fp32_equivalent_bytes': fp32_bytes,
            'compress_activations': self.compress_activations,
            'activation_compression_ratio': 0.5 if self.compress_activations else 1.0,
        }

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, threshold={self.threshold}, TC=True')


class Int2LinearTCWithGradHook(Int2LinearTC):
    """
    Int2LinearTC with automatic gradient capture via hooks.

    Inherits INT8 activation compression from Int2LinearTC (Phase 2).
    """

    def __init__(self, *args, compress_activations: bool = True, **kwargs):
        super().__init__(*args, compress_activations=compress_activations, **kwargs)
        self._lr = 0.001

    def set_lr(self, lr: float):
        """Update the learning rate for hysteresis updates."""
        self._lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)

        if self.training and out.requires_grad:
            out.register_hook(self._grad_hook)

        return out

    def _grad_hook(self, grad: torch.Tensor):
        """Hook called during backward to capture output gradient."""
        if self._saved_input is not None:
            self.hysteresis_update(grad, self._lr)


def convert_int2_to_tc(module: nn.Module, compress_activations: bool = True) -> nn.Module:
    """
    Convert all Int2Linear layers in a module to Int2LinearTC.

    Args:
        module: PyTorch module containing Int2Linear layers
        compress_activations: Enable INT8 activation compression (Phase 2)

    Returns:
        Module with Int2Linear replaced by Int2LinearTC
    """
    # Import original Int2Linear for isinstance check
    from int2_linear import Int2Linear, Int2LinearWithGradHook

    for name, child in module.named_children():
        if isinstance(child, Int2LinearWithGradHook):
            # Convert with grad hook
            tc_layer = Int2LinearTCWithGradHook(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                threshold=child.threshold,
                lr_scale=child.lr_scale,
                decay_rate=child.decay_rate,
                compress_activations=compress_activations
            )
            tc_layer.W_packed.copy_(child.W_packed)
            tc_layer.H_packed.copy_(child.H_packed)
            tc_layer.gamma.data.copy_(child.gamma.data)
            if child.bias is not None:
                tc_layer.bias.data.copy_(child.bias.data)
            tc_layer._step.copy_(child._step)
            tc_layer._initialized = child._initialized
            tc_layer._lr = child._lr
            setattr(module, name, tc_layer)

        elif isinstance(child, Int2Linear):
            # Convert basic layer
            tc_layer = Int2LinearTC(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                threshold=child.threshold,
                lr_scale=child.lr_scale,
                decay_rate=child.decay_rate,
                compress_activations=compress_activations
            )
            tc_layer.W_packed.copy_(child.W_packed)
            tc_layer.H_packed.copy_(child.H_packed)
            tc_layer.gamma.data.copy_(child.gamma.data)
            if child.bias is not None:
                tc_layer.bias.data.copy_(child.bias.data)
            tc_layer._step.copy_(child._step)
            tc_layer._initialized = child._initialized
            setattr(module, name, tc_layer)
        else:
            # Recurse
            convert_int2_to_tc(child, compress_activations=compress_activations)

    return module
