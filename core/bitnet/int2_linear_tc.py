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
        name=f"int2_tc_ops_consolidated_v5_{_CUDA_ARCH.replace('-arch=', '')}",
        sources=[
            os.path.join(KERNELS_DIR, "int2_tc_ops.cpp"),
            os.path.join(KERNELS_DIR, "int2_matmul_tc.cu"),
            os.path.join(KERNELS_DIR, "int2_matmul_int8_tc_v2.cu"),
            os.path.join(KERNELS_DIR, "int2_backward_tc.cu"),
            os.path.join(KERNELS_DIR, "int2_backward_int8_tc_v2.cu"),
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


def apply_24_sparsity(W_ternary: torch.Tensor) -> torch.Tensor:
    """Forza sparsità 2:4 su pesi ternari {-1,0,+1}.

    In ogni gruppo di 4 elementi, mantieni esattamente i 2 con valore assoluto
    più alto; azzera gli altri. Con pesi ternari il risultato è sempre in {-1,0,+1}.

    Args:
        W_ternary: tensor float o INT8 ternario [N, K], valori in {-1, 0, 1}

    Returns:
        Tensor con sparsità 2:4, stessi shape e dtype.
    """
    original_dtype = W_ternary.dtype
    N, K = W_ternary.shape

    # Pad K a multiplo di 4 se necessario
    pad = (4 - K % 4) % 4
    if pad > 0:
        W_padded = torch.nn.functional.pad(W_ternary.float(), (0, pad))
    else:
        W_padded = W_ternary.float()

    K_padded = W_padded.shape[1]
    W_grouped = W_padded.view(N, K_padded // 4, 4)  # [N, groups, 4]

    # Top-2 per gruppo (per valore assoluto)
    abs_vals = W_grouped.abs()
    _, top2_idx = abs_vals.topk(2, dim=2)

    mask = torch.zeros_like(W_grouped)
    mask.scatter_(2, top2_idx, 1.0)

    W_sparse = (W_grouped * mask).view(N, K_padded)

    # Rimuovi padding
    if pad > 0:
        W_sparse = W_sparse[:, :K]

    return W_sparse.to(dtype=original_dtype)


# ===========================================================================
# BitNet v2: Walsh-Hadamard Transform + INT4 Quantization
# ===========================================================================

def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard Transform sull'ultima dimensione.

    x.shape[-1] deve essere potenza di 2.
    Normalizza per 1/sqrt(K) → trasformazione ortonormale.
    """
    K = x.shape[-1]
    assert K > 0 and (K & (K - 1)) == 0, f"Last dim must be power of 2, got {K}"

    result = x.clone()
    step = 1
    while step < K:
        # Butterfly: pair each element with its partner at distance `step`
        n_groups = K // (2 * step)
        r = result.view(*result.shape[:-1], n_groups, 2, step)
        a = r[..., 0, :].clone()  # clone to avoid in-place issues
        b = r[..., 1, :].clone()
        r_new = torch.stack([a + b, a - b], dim=-2)
        result = r_new.view(*result.shape)
        step <<= 1

    return result / (K ** 0.5)


def quantize_hadamard_int4(x: torch.Tensor):
    """Hadamard + INT4 quantization per le attivazioni.

    Applica WHT per distribuire gli outlier, poi quantizza a INT4 [-7,7].
    Risparmio memoria: -50% rispetto a INT8.

    Args:
        x: Tensor [*, K] (float16 o bfloat16), K deve essere potenza di 2

    Returns:
        packed: Tensor [*, K//2] uint8 (2 INT4 per byte)
        scale: Tensor [*] float32 (scala per riga)
    """
    orig_shape = x.shape
    K = x.shape[-1]

    # Pad K a prossima potenza di 2 se necessario
    K_padded = 1 << (K - 1).bit_length() if K > 1 else 1

    # Lavora in float32 per precisione
    x_flat = x.reshape(-1, K).float()
    M = x_flat.shape[0]

    if K_padded != K:
        x_flat = torch.nn.functional.pad(x_flat, (0, K_padded - K))

    # Hadamard transform (ortonormale)
    x_h = _hadamard_transform(x_flat)  # [M, K_padded]

    if K_padded != K:
        x_h = x_h[:, :K]

    # Per-row scale: absmax / 7
    absmax = x_h.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [M, 1]
    scale = absmax / 7.0

    # Quantize a INT4 [-7, 7]
    x_q = (x_h / scale).round().clamp(-7, 7).to(torch.int8)  # [M, K]

    # Shift a [0, 14] per nibble encoding senza segno
    x_shifted = x_q + 7  # [0, 14], ancora int8

    # Pad K a multiplo di 2
    if K % 2 != 0:
        x_shifted = torch.nn.functional.pad(x_shifted, (0, 1))

    # Pack: byte0 = low_nibble | (high_nibble << 4)
    even = x_shifted[:, 0::2].to(torch.uint8)  # [M, K//2]
    odd  = x_shifted[:, 1::2].to(torch.uint8)  # [M, K//2]
    packed = (even & 0x0F) | ((odd & 0x0F) << 4)  # [M, K//2]

    # Reshape
    out_k = packed.shape[-1]
    out_shape_packed = list(orig_shape[:-1]) + [out_k]
    out_shape_scale  = list(orig_shape[:-1])

    return packed.view(out_shape_packed), scale.squeeze(-1).view(out_shape_scale)


def dequantize_hadamard_int4(packed: torch.Tensor, scale: torch.Tensor, K: int) -> torch.Tensor:
    """Inverso: unpack INT4 + WHT inverso.

    Args:
        packed: [*, K//2] uint8
        scale:  [*] float32
        K:      dimensione originale delle attivazioni

    Returns:
        [*, K] float16
    """
    orig_shape_scale = scale.shape
    M = packed.numel() // packed.shape[-1]
    K_half = packed.shape[-1]

    packed_flat = packed.reshape(M, K_half)
    scale_flat  = scale.reshape(M, 1).float()

    # Unpack nibbles
    low  = (packed_flat & 0x0F).to(torch.int8)           # [M, K_half]
    high = ((packed_flat >> 4) & 0x0F).to(torch.int8)    # [M, K_half]

    # Shift da [0,14] a [-7,7]
    low  = low  - 7
    high = high - 7

    # Interleave: [even, odd] → [M, 2*K_half]
    x_int4 = torch.empty(M, 2 * K_half, dtype=torch.int8, device=packed.device)
    x_int4[:, 0::2] = low
    x_int4[:, 1::2] = high

    # Tronca a K se c'era padding
    if 2 * K_half > K:
        x_int4 = x_int4[:, :K]

    # Dequantize
    x_float = x_int4.float() * scale_flat  # [M, K]

    # Inverse Hadamard: applica WHT di nuovo (H @ H = K*I, normalizzato 1/sqrt(K))
    K_padded = 1 << (K - 1).bit_length() if K > 1 else 1
    if K_padded != K:
        x_float = torch.nn.functional.pad(x_float, (0, K_padded - K))

    x_rec = _hadamard_transform(x_float)

    if K_padded != K:
        x_rec = x_rec[:, :K]

    # Reshape a forma originale
    out_shape = list(orig_shape_scale) + [K]
    return x_rec.half().view(out_shape)


class Int2LinearTCFunction(Function):
    """Custom autograd function using TC-optimized kernels (INT8 IMMA path)."""

    # Shared absmax buffer for async quantization — avoids cudaStreamSynchronize.
    # Dict keyed by device to support single-process multi-GPU if needed.
    _absmax_bufs: dict = {}

    @staticmethod
    def _get_absmax_buf(device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in Int2LinearTCFunction._absmax_bufs:
            Int2LinearTCFunction._absmax_bufs[key] = torch.zeros(
                1, dtype=torch.float32, device=device
            )
        return Int2LinearTCFunction._absmax_bufs[key]

    @staticmethod
    def forward(ctx, X: torch.Tensor, W_packed: torch.Tensor, gamma: torch.Tensor,
                K: int, gamma_cached: float = None) -> torch.Tensor:
        """
        Forward pass: Y = X_int8 @ W_int8.T * (scale_x/127) * gamma
        Uses INT8 IMMA Tensor Cores (2x throughput vs FP16 WMMA).
        """
        ctx.save_for_backward(W_packed, gamma)
        ctx.K = K
        ctx.gamma_cached = gamma_cached

        X_half = X.half() if X.dtype != torch.float16 else X
        gamma_val = gamma_cached if gamma_cached is not None else gamma.item()

        # Quantize X to INT8 asynchronously (no GPU→CPU sync)
        d_absmax = Int2LinearTCFunction._get_absmax_buf(X.device)
        X_int8, d_scale_x = int2_tc_ops.quantize_activation_async(
            X_half.contiguous(), d_absmax
        )

        # INT8 matmul: Y = X_int8 @ W_int8.T * (scale_x/127) * gamma
        Y = int2_tc_ops.matmul_int8(X_int8, W_packed, d_scale_x, gamma_val)

        return Y

    @staticmethod
    def backward(ctx, dY: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass: dX = dY_int8 @ W_int8 * (scale_dy/127) * gamma
        Uses INT8 IMMA Tensor Cores.
        """
        W_packed, gamma = ctx.saved_tensors
        K = ctx.K

        dX = None
        if ctx.needs_input_grad[0]:
            dY_half = dY.half() if dY.dtype != torch.float16 else dY
            gamma_val = ctx.gamma_cached if ctx.gamma_cached is not None else gamma.item()

            # Quantize dY to INT8 asynchronously
            d_absmax = Int2LinearTCFunction._get_absmax_buf(dY.device)
            dY_int8, d_scale_dy = int2_tc_ops.quantize_activation_async(
                dY_half.contiguous(), d_absmax
            )

            # INT8 backward: dX = dY_int8 @ W_int8 * (scale_dy/127) * gamma
            dX = int2_tc_ops.backward_input_int8(
                dY_int8, W_packed, d_scale_dy, gamma_val, K
            )

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
        compress_activations: bool = True,  # Phase 2: Enable INT8 compression
        use_24_sparsity: bool = False,  # Sparse-BitNet: 2:4 structured sparsity
        use_hadamard_int4: bool = False    # BitNet v2: Hadamard+INT4 (vs INT8)
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
        self.use_24_sparsity = use_24_sparsity
        self.use_hadamard_int4 = use_hadamard_int4
        # Pre-alloca K_padded per Hadamard (potenza di 2 >= in_features)
        if use_hadamard_int4:
            k = in_features
            k_pad = 1 << (k - 1).bit_length() if k > 1 else 1
            self.register_buffer('_had_K_padded', torch.tensor(k_pad, dtype=torch.int64))

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

        # Sparse-BitNet: applica sparsità 2:4 prima del packing
        if self.use_24_sparsity:
            w_ternary = apply_24_sparsity(w_ternary)

        # Always use CPU packing (fast enough for initialization)
        self._pack_int2_cpu(w_ternary)

        self.gamma.data.fill_(scale)
        self._gamma_cached = scale  # Cache initial gamma
        self.H_packed.zero_()
        self._initialized = True

    def _pack_int2_cpu(self, w_ternary: torch.Tensor):
        """Vectorized CPU packing (no Python loops)."""
        import numpy as np
        N, K = w_ternary.shape
        packed_K = (K + 3) // 4

        w_cpu = w_ternary.cpu().numpy().astype(np.int8)
        # Encode: {-1,0,+1} → {0,1,2}
        encoded = (w_cpu + 1).astype(np.uint8)

        # Pad K to multiple of 4
        pad = packed_K * 4 - K
        if pad > 0:
            encoded = np.pad(encoded, ((0, 0), (0, pad)), constant_values=0)

        # Reshape to (N, packed_K, 4) and pack 4 values per byte
        encoded = encoded.reshape(N, packed_K, 4)
        packed_cpu = (encoded[:, :, 0] |
                      (encoded[:, :, 1] << 2) |
                      (encoded[:, :, 2] << 4) |
                      (encoded[:, :, 3] << 6))

        self.W_packed.copy_(torch.from_numpy(packed_cpu.astype(np.uint8)).to(w_ternary.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using TC-optimized matmul."""
        if not self._initialized:
            w_init = torch.randn(
                self.out_features, self.in_features,
                device=x.device, dtype=torch.float32
            ) * (1.0 / math.sqrt(self.in_features))
            self.init_from_float(w_init)

        if self.training:
            x_detached = x.detach()
            if self.use_hadamard_int4:
                # BitNet v2: Hadamard + INT4 compression (50% less memory than INT8)
                x_half = x_detached.half() if x_detached.dtype != torch.float16 else x_detached
                self._saved_input_shape = x_half.shape
                packed, scale = quantize_hadamard_int4(x_half.contiguous())
                self._saved_input = (packed, scale, 'int4_hadamard')
            elif self.compress_activations:
                # Phase 2: INT8 compression
                x_half = x_detached.half() if x_detached.dtype != torch.float16 else x_detached
                self._saved_input_shape = x_half.shape
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

        # Decomprimi attivazione salvata
        if isinstance(self._saved_input, tuple) and len(self._saved_input) == 3 and self._saved_input[2] == 'int4_hadamard':
            # BitNet v2: Hadamard INT4 decompressione
            packed, scale, _ = self._saved_input
            K = self._saved_input_shape[-1] if self._saved_input_shape is not None else self.in_features
            X = dequantize_hadamard_int4(packed, scale, K)
            if self._saved_input_shape is not None:
                X = X.view(self._saved_input_shape)
        elif self.compress_activations and isinstance(self._saved_input, tuple):
            x_int8, scale = self._saved_input
            X = int2_tc_ops.dequantize_activation(x_int8, scale)
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

        # Sparse-BitNet: applica sparsità 2:4 dopo hysteresis update
        if self.use_24_sparsity:
            # Decomprimi W_packed → FP16 usando il kernel unpack_int2
            W_fp16 = torch.empty(self.out_features, self.in_features,
                                 dtype=torch.float16, device=self.W_packed.device)
            int2_tc_ops.unpack_int2(self.W_packed, W_fp16, self.in_features)
            # Applica maschera 2:4 (lavora su float, risultato in {-1,0,+1})
            W_sparse_fp16 = apply_24_sparsity(W_fp16)
            # Riconverti a INT8 e repack
            W_sparse_int8 = W_sparse_fp16.round().clamp(-1, 1).to(torch.int8).cpu()
            self._pack_int2_cpu(W_sparse_int8)

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

    def __init__(self, *args, compress_activations: bool = True, use_24_sparsity: bool = False, use_hadamard_int4: bool = False, **kwargs):
        super().__init__(*args, compress_activations=compress_activations, use_24_sparsity=use_24_sparsity, use_hadamard_int4=use_hadamard_int4, **kwargs)
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
