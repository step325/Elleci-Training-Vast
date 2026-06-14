"""
Task 5 — BitNet v2 (Hadamard + INT4 Activations)
Valida: _hadamard_transform (involutoria), quantize_hadamard_int4 / dequantize_hadamard_int4
(shape, range, round-trip RMS error < 5%).
"""
import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.bitnet.int2_linear_tc import (
    _hadamard_transform,
    quantize_hadamard_int4,
    dequantize_hadamard_int4,
)


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform
# ---------------------------------------------------------------------------

def test_wht_involutory():
    """H(H(x)) = x — la trasformazione normalizzata è involutoria."""
    torch.manual_seed(0)
    x = torch.randn(4, 16)  # K=16, potenza di 2

    x_hht = _hadamard_transform(_hadamard_transform(x))
    assert torch.allclose(x_hht, x, atol=1e-5), (
        f"WHT non involutoria: max_err={( x_hht - x).abs().max().item():.2e}"
    )


def test_wht_shape_preserved():
    """La WHT preserva la shape dell'input."""
    for shape in [(8, 32), (16, 64), (4, 4)]:
        x = torch.randn(*shape)
        out = _hadamard_transform(x)
        assert out.shape == x.shape, f"Shape cambiata: {x.shape} → {out.shape}"


def test_wht_orthogonal():
    """La WHT è una trasformazione ortonormale: ||H(x)||^2 = ||x||^2."""
    x = torch.randn(8, 64)
    x_h = _hadamard_transform(x)
    assert torch.allclose(x.norm(), x_h.norm(), atol=1e-4), (
        f"Norma non preservata: ||x||={x.norm():.4f}, ||H(x)||={x_h.norm():.4f}"
    )


# ---------------------------------------------------------------------------
# quantize_hadamard_int4
# ---------------------------------------------------------------------------

def test_packed_shape():
    """Output shape: packed=[M, K//2], scale=[M]."""
    x = torch.randn(8, 256).half().cuda()
    packed, scale = quantize_hadamard_int4(x)

    assert packed.shape == (8, 128), f"packed.shape={packed.shape}, atteso (8,128)"
    assert scale.shape == (8,), f"scale.shape={scale.shape}, atteso (8,)"
    assert packed.dtype == torch.uint8, f"packed.dtype={packed.dtype}"


def test_packed_shape_batch():
    """Funziona su batch 3D: [B, T, K] → packed [B, T, K//2]."""
    x = torch.randn(2, 16, 128).half().cuda()
    packed, scale = quantize_hadamard_int4(x)

    assert packed.shape == (2, 16, 64)
    assert scale.shape == (2, 16)


def test_int4_range():
    """I valori INT4 estratti dal packed devono stare in [-7, 7]."""
    x = torch.randn(16, 128).half().cuda()
    packed, _ = quantize_hadamard_int4(x)

    packed_flat = packed.reshape(-1, packed.shape[-1])
    low  = (packed_flat & 0x0F).to(torch.int8) - 7
    high = ((packed_flat >> 4) & 0x0F).to(torch.int8) - 7

    assert low.abs().max().item() <= 7, f"low max abs={low.abs().max().item()}"
    assert high.abs().max().item() <= 7, f"high max abs={high.abs().max().item()}"


# ---------------------------------------------------------------------------
# dequantize_hadamard_int4 — round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_rms():
    """Round-trip (quant → dequant) deve avere errore RMS relativo < 20%.

    INT4 ha 15 livelli: l'errore teorico di quantizzazione per segnale gaussiano
    è ~12-15% relativo. La soglia 20% lascia margine per variazioni statistiche.
    """
    torch.manual_seed(42)
    x = torch.randn(64, 512).half().cuda()

    packed, scale = quantize_hadamard_int4(x)
    K = x.shape[-1]
    x_rec = dequantize_hadamard_int4(packed, scale, K).half()

    rms_error  = (x.float() - x_rec.float()).pow(2).mean().sqrt()
    rms_signal = x.float().pow(2).mean().sqrt()
    relative_error = (rms_error / (rms_signal + 1e-8)).item()

    assert relative_error < 0.20, (
        f"Round-trip RMS error={relative_error:.4f} (>= 20%)"
    )


def test_roundtrip_no_nan():
    """Nessun NaN/Inf dopo il round-trip."""
    x = torch.randn(32, 256).half().cuda()
    packed, scale = quantize_hadamard_int4(x)
    K = x.shape[-1]
    x_rec = dequantize_hadamard_int4(packed, scale, K)

    assert not torch.isnan(x_rec).any(), "NaN nel risultato del round-trip"
    assert not torch.isinf(x_rec).any(), "Inf nel risultato del round-trip"


def test_roundtrip_shape():
    """La forma del tensore ricostruito deve essere uguale all'originale."""
    x = torch.randn(4, 8, 64).half().cuda()
    packed, scale = quantize_hadamard_int4(x)
    K = x.shape[-1]
    x_rec = dequantize_hadamard_int4(packed, scale, K)

    assert x_rec.shape == x.shape, f"Shape dopo round-trip: {x_rec.shape} != {x.shape}"
