
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import warnings


# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CUDA_SRC_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "../../kernels/mamba2"))

sources = [
    os.path.join(CUDA_SRC_DIR, "mamba2_ssd.cpp"),
    os.path.join(CUDA_SRC_DIR, "mamba2_ssd_kernel.cu"),
]

# JIT Compile / Load
# verbose=True to see compilation errors
try:
    mamba2_ssd_cuda = load(
        name="mamba2_ssd_cuda",
        sources=sources,
        verbose=True,
        extra_cuda_cflags=["-O3"]
    )
    print("✅ Mamba-2 SSD Custom Kernel Loaded Successfully")
except Exception as e:
    print(f"❌ Failed to load Mamba-2 SSD Kernel: {e}")
    mamba2_ssd_cuda = None

from .mamba2_optimized import Mamba2BlockMatmul, DifferentialMamba2BlockMatmul

class Mamba2SSDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C):
        # x: [B, NC, CS, H, D]
        # dt: [B, NC, CS, H]
        # A: [H]
        # B, C: [B, NC, CS, P]
        
        # Ensure contiguous
        x = x.contiguous()
        dt = dt.contiguous()
        A = A.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        
        y = mamba2_ssd_cuda.forward(x, dt, A, B, C)
        
        ctx.save_for_backward(x, dt, A, B, C)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, dt, A, B, C = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        
        # M1 risolto: il backward (mamba2_ssd.cpp) calcola dx/ddt/dA/dB/dC corretti
        # via tensor-ops (verificato con gradcheck fp64, relerr ~1e-16).
        dx, ddt, dA, dB, dC = mamba2_ssd_cuda.backward(grad_output, x, dt, A, B, C)
        return dx, ddt, dA, dB, dC

# ---------------------------------------------------------------------------
# Dominio di validità del kernel CUDA SSD (M3)
# ---------------------------------------------------------------------------
# Il kernel fuso (mamba2_ssd_kernel.cu) calcola SOLO il contributo intra-chunk
# (blocco diagonale) e tratta dt come per-head (usa dt[...,0]). NON propaga lo
# stato inter-chunk e NON supporta dt per-canale. È quindi corretto solo per
# seqlen <= chunk_size con dt per-head. Per il training reale (seqlen >>
# chunk_size, dt = dt_proj per canale) usare use_matmul_ssd / use_checkpointing.
_CUDA_SSD_DT_WARNED = False


def _check_cuda_ssd_domain(seqlen: int, chunk_size: int, dt: torch.Tensor) -> None:
    """Fa fallire/avvisare quando il path CUDA SSD esce dal suo dominio corretto."""
    global _CUDA_SSD_DT_WARNED
    n_chunks = (seqlen + chunk_size - 1) // chunk_size
    if n_chunks > 1:
        raise NotImplementedError(
            f"Mamba2 CUDA SSD: il kernel calcola solo il contributo INTRA-chunk "
            f"(nessuno stato inter-chunk). seqlen={seqlen} > chunk_size={chunk_size} "
            f"→ {n_chunks} chunk: i risultati sarebbero ERRATI. Disattiva use_cuda_ssd "
            f"e usa use_matmul_ssd o use_checkpointing."
        )
    if dt.dim() == 4 and dt.shape[-1] > 1 and not _CUDA_SSD_DT_WARNED:
        # dt_proj mappa n_heads→d_inner: dt è per-canale. Il kernel userà dt[...,0]
        # (per-head) → perdita d'informazione se i canali differiscono.
        _CUDA_SSD_DT_WARNED = True
        if not torch.allclose(dt[..., :1].expand_as(dt), dt, rtol=1e-3, atol=1e-3):
            warnings.warn(
                "Mamba2 CUDA SSD: dt è per-canale (d_inner) ma il kernel usa dt[...,0] "
                "(per-head) → risultati approssimati. Preferire use_matmul_ssd.",
                RuntimeWarning, stacklevel=2,
            )


class Mamba2BlockCuda(Mamba2BlockMatmul):
    """
    Uses the Custom CUDA Kernel for the chunk_forward pass.
    Drastically reduces VRAM by recomputing 'M' in the backward pass.

    ATTENZIONE (M3): valido solo per seqlen <= chunk_size con dt per-head.
    Vedi _check_cuda_ssd_domain. Per il training reale usare use_matmul_ssd.
    """
    
    def chunked_ssd(self, x, dt, A, B, C, initial_state=None):
        """
        Fused Chunked SSD using Custom CUDA Kernel.
        Processes all chunks in parallel (one kernel launch).
        
        Args:
            x: [B, L, H, D] (Padded to multiple of chunk_size)
            dt: [B, L, H] (Padded) -> Kernel assumes [B, NC, CS, H] so we need expand? 
                Wait, kernel takes dt [B, NC, CS, H].
                Input dt here is [B, L, H].
            A: [H] (Unused here, self.A_log used) but passed as arg?
               Mamba2Block.chunked_ssd source says: chunked_ssd(x_heads, dt_heads, A, B, C)
               So A is passed.
            B: [B, L, N]
            C: [B, L, N]
        """
        from einops import rearrange
        
        cs = self.chunk_size
        batch, seqlen, n_heads, head_dim = x.shape
        # Input seqlen is already padded to multiple of cs by method calling this
        assert seqlen % cs == 0, f"Seqlen {seqlen} must be multiple of chunk_size {cs}"
        _check_cuda_ssd_domain(seqlen, cs, dt)  # M3: intra-chunk + dt per-head only

        # Reshape to 5D for Kernel: [B, NC, CS, H, D]
        x_5d = rearrange(x, 'b (nc cs) h d -> b nc cs h d', cs=cs)
        
        # dt: [B, L, H, D] -> [B, NC, CS, H, D] (Wait, kernel takes [B, NC, CS, H]??)
        # CHECK KERNEL: const scalar_t* dt, // [B, n_chunks, CS, H]
        # Kernel expects dt to be 4D: [B, NC, CS, H].
        # But input dt is [B, L, H, D].
        # Does dt vary across D? Usually NO. dt is broadcasted over D.
        # If dt has shape [B, L, H, D], checks if D is 1? Or if values are identical?
        # Mamba-2 usually has dt per head.
        # If D>1, maybe I need to slice it? dt[..., 0]?
        # Let's check mamba2.py again.
        # dt_chunks = rearrange(dt, 'b (nc cs) h d -> b nc cs h d', ...)
        # dt_cumsum = dt_chunks.cumsum(dim=2)
        # S = A * dt_cumsum
        # If dt has D, then S has D.
        
        # My Kernel signature:
        # const scalar_t* __restrict__ dt, // [B, n_chunks, CS, H]
        #
        # My forward kernel does: float dt_val = (float)dt[dt_offset + r * stride_dt_cs + h];
        # It assumes dt is per head (H), not per D.
        # So I expect dt to be [B, NC, CS, H].
        
        # If the input `dt` from `Mamba2Block` is [B, L, H, D], I must verify if D=1.
        # If D > 1, I might need to adapt the kernel OR slice it here.
        # Usually in Mamba-2, dt is generated as [B, L, n_heads]. 
        # But `mamba2.py` in this repo seems to produce [B, L, H, D]?
        
        # Let's handle the mismatch here.
        # If dt is 4D, check last dim size.
        if dt.dim() == 4:
             # Assume D=1 or we just take the first element if D>1 (assuming broadcast)
             # But if D varies, my kernel is wrong.
             # Standard Mamba2: dt projection is to n_heads. Broadcast to D.
             # So dt[..., 0] is sufficient.
             dt = dt[..., 0] # Now [B, L, H]
        
        dt_5d = rearrange(dt, 'b (nc cs) h -> b nc cs h', cs=cs)
        
        # B, C: [B, L, N] -> [B, NC, CS, N]
        B_5d = rearrange(B, 'b (nc cs) n -> b nc cs n', cs=cs)
        C_5d = rearrange(C, 'b (nc cs) n -> b nc cs n', cs=cs)
        
        # A parameter (Raw A_log used inside kernel? No, kernel expects A)
        # Kernel: float a_val = -expf((float)A[h]);
        # If input 'A' is A_log, this is correct.
        # Check source of A passed to chunked_ssd:
        # In ssd_forward: A = self.A_log
        # So 'A' here is indeed A_log.
        # Accessing A from args instead of self.A_log to support potential overrides
        A_log = A 
        
        # Run Kernel
        y_5d = Mamba2SSDFunction.apply(x_5d, dt_5d, A_log, B_5d, C_5d)
        
        # Reshape output back: [B, NC, CS, H, D] -> [B, L, H, D]
        y = rearrange(y_5d, 'b nc cs h d -> b (nc cs) h d')
        
        return y


class DifferentialMamba2BlockCuda(DifferentialMamba2BlockMatmul):
    def chunked_ssd(self, x, dt, A, B, C, initial_state=None):
        """
        Fused Chunked SSD using Custom CUDA Kernel.
        Processes all chunks in parallel (one kernel launch).
        """
        from einops import rearrange
        
        cs = self.chunk_size
        batch, seqlen, n_heads, head_dim = x.shape
        assert seqlen % cs == 0, f"Seqlen {seqlen} must be multiple of chunk_size {cs}"
        _check_cuda_ssd_domain(seqlen, cs, dt)  # M3: intra-chunk + dt per-head only

        # Reshape to 5D for Kernel: [B, NC, CS, H, D]
        x_5d = rearrange(x, 'b (nc cs) h d -> b nc cs h d', cs=cs)
        if dt.dim() == 4:
             dt = dt[..., 0]
        
        dt_5d = rearrange(dt, 'b (nc cs) h -> b nc cs h', cs=cs).to(x.dtype)
        B_5d = rearrange(B, 'b (nc cs) n -> b nc cs n', cs=cs).to(x.dtype)
        C_5d = rearrange(C, 'b (nc cs) n -> b nc cs n', cs=cs).to(x.dtype)
        
        # Debug Dtypes
        # print(f"DEBUG Mamba2Cuda: x={x.dtype} dt={dt.dtype} A={A.dtype} B={B.dtype} C={C.dtype}")
        
        # A_log passed as arg 'A'
        A_log = A.to(x.dtype)
        
        # Run Kernel
        y_5d = Mamba2SSDFunction.apply(x_5d, dt_5d, A_log, B_5d, C_5d)
        
        # Reshape output back: [B, NC, CS, H, D] -> [B, L, H, D]
        y = rearrange(y_5d, 'b nc cs h d -> b (nc cs) h d')
        
        return y
