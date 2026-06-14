"""
JIT-build degli stessi sorgenti CUDA caricati dal modello + utility di
packing/reference INT2 condivise dai test.

Usa nomi di estensione dedicati (`*_audit`) per forzare una build pulita
dei sorgenti correnti, indipendente dalla cache del modello.
"""
import os

import torch
from torch.utils.cpp_extension import load

KERNELS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INT2_DIR = os.path.join(KERNELS_DIR, "int2")
MAMBA2_DIR = os.path.join(KERNELS_DIR, "mamba2")

# Encoding INT2 (vedi int2_packed.cuh): 00=-1, 01=0, 10=+1, 11->0
_DECODE_LUT = torch.tensor([-1, 0, 1, 0], dtype=torch.int8)


def cuda_arch_flag() -> str:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"-arch=sm_{major}{minor}"
    return "-arch=sm_80"


def build_int2():
    """Compila l'estensione INT2 consolidata (9 sorgenti) e la restituisce."""
    arch = cuda_arch_flag()
    return load(
        name=f"int2_tc_ops_audit_{arch.replace('-arch=', '')}",
        sources=[
            os.path.join(INT2_DIR, "int2_tc_ops.cpp"),
            os.path.join(INT2_DIR, "int2_matmul_tc.cu"),
            os.path.join(INT2_DIR, "int2_matmul_int8_tc_v2.cu"),
            os.path.join(INT2_DIR, "int2_backward_tc.cu"),
            os.path.join(INT2_DIR, "int2_backward_int8_tc_v2.cu"),
            os.path.join(INT2_DIR, "int2_hysteresis_tc.cu"),
            os.path.join(INT2_DIR, "int2_hysteresis_v2.cu"),
            os.path.join(INT2_DIR, "int2_activation_quant.cu"),
            os.path.join(INT2_DIR, "int2_unpack.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", arch],
        verbose=True,
    )


def build_mamba2():
    """Compila l'estensione Mamba-2 SSD e la restituisce."""
    arch = cuda_arch_flag()
    return load(
        name=f"mamba2_ssd_audit_{arch.replace('-arch=', '')}",
        sources=[
            os.path.join(MAMBA2_DIR, "mamba2_ssd.cpp"),
            os.path.join(MAMBA2_DIR, "mamba2_ssd_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", arch],
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Reference INT2 (PyTorch puro) — devono combaciare bit-per-bit col decode CUDA
# ---------------------------------------------------------------------------

def pack_int2(w_ternary: torch.Tensor) -> torch.Tensor:
    """[N, K] in {-1,0,1} -> [N, packed_K] uint8 (encoding del kernel)."""
    n, k = w_ternary.shape
    packed_k = (k + 3) // 4
    enc = (w_ternary.clamp(-1, 1) + 1).to(torch.uint8)  # -1->0, 0->1, +1->2
    pad = packed_k * 4 - k
    if pad:
        enc = torch.cat([enc, torch.ones(n, pad, dtype=torch.uint8, device=w_ternary.device)], dim=1)
    enc = enc.view(n, packed_k, 4)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=w_ternary.device)
    packed = (enc << shifts).sum(dim=2).to(torch.uint8)
    return packed


def unpack_int2(w_packed: torch.Tensor, k: int) -> torch.Tensor:
    """[N, packed_K] uint8 -> [N, K] float (decode del kernel)."""
    n, pk = w_packed.shape
    lut = _DECODE_LUT.to(w_packed.device)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=w_packed.device)
    fields = (w_packed.unsqueeze(-1) >> shifts) & 0x3  # [N, pk, 4]
    vals = lut[fields.long()]  # [N, pk, 4]
    return vals.reshape(n, pk * 4)[:, :k].float()


def make_packed_weights(n: int, k: int, device="cuda", seed=0):
    """Restituisce (W_packed uint8, W_ternary float) coerenti tra loro."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randint(-1, 2, (n, k), generator=g, device=device, dtype=torch.int8).float()
    return pack_int2(w), w
