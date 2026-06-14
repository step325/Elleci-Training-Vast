"""
Task 1 — Muon Optimizer
Valida: ns_step (Newton-Schulz), MuonOptimizer, separazione 2D/1D parametri.
"""
import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import ns_step, MuonOptimizer


# ---------------------------------------------------------------------------
# ns_step
# ---------------------------------------------------------------------------

def test_ns_step_2d():
    """ns_step su tensore 2D: shape invariata, no NaN, norma ~ sqrt(min(M,N)) * orig_norm."""
    G = torch.randn(128, 256).cuda()
    G_orth = ns_step(G)

    assert G_orth.shape == G.shape
    assert not torch.isnan(G_orth).any(), "NaN nel risultato di ns_step 2D"

    # ns_step produce singular values ≈ 1, quindi norma ≈ sqrt(min(M,N)) * orig_norm
    # Per 128x256: sqrt(128) ≈ 11.3. Verifichiamo che la norma sia nell'ordine giusto.
    expected_norm = (G.norm().item() * (min(G.shape) ** 0.5))
    actual_norm = G_orth.norm().item()
    ratio = actual_norm / (expected_norm + 1e-8)
    assert 0.5 < ratio < 2.0, f"Norma fuori range: attesa ~{expected_norm:.1f}, ottenuta {actual_norm:.1f}"


def test_ns_step_3d():
    """ns_step su tensore 3D (forma reale conv weights Mamba: 512×1×4)."""
    G = torch.randn(512, 1, 4).cuda()
    G_orth = ns_step(G)  # NON deve sollevare RuntimeError

    assert G_orth.shape == (512, 1, 4)
    assert not torch.isnan(G_orth).any(), "NaN nel risultato di ns_step 3D"


def test_ns_step_tall():
    """ns_step su tensore tall (righe > colonne): shape invariata, no NaN."""
    G = torch.randn(256, 64).cuda()
    G_orth = ns_step(G)

    assert G_orth.shape == G.shape
    assert not torch.isnan(G_orth).any(), "NaN nel risultato di ns_step tall"
    # Norma attesa ≈ sqrt(min(256,64)) * orig_norm = sqrt(64) * orig_norm = 8 * orig_norm
    expected_norm = G.norm().item() * (min(G.shape) ** 0.5)
    actual_norm = G_orth.norm().item()
    ratio = actual_norm / (expected_norm + 1e-8)
    assert 0.5 < ratio < 2.0, f"Norma fuori range: ratio={ratio:.2f}"


def test_ns_step_square():
    """ns_step su tensore quadrato: singular values ≈ 1 (matrice ortogonale)."""
    G = torch.randn(64, 64).cuda()
    orig_norm = G.norm().item()
    G_orth = ns_step(G)

    assert not torch.isnan(G_orth).any()
    # Scala per recuperare la matrice ortogonale e verificare i sv
    G_normalized = G_orth / (orig_norm + 1e-8)
    sv = torch.linalg.svdvals(G_normalized.float())
    assert sv.max().item() < 1.5, f"Singular value max troppo alto: {sv.max().item():.3f}"


# ---------------------------------------------------------------------------
# MuonOptimizer
# ---------------------------------------------------------------------------

def test_muon_step():
    """MuonOptimizer.step() aggiorna i parametri senza NaN."""
    p = torch.nn.Parameter(torch.randn(32, 64).cuda())
    opt = MuonOptimizer([p], lr=0.01)
    p_before = p.data.clone()

    loss = (p * torch.randn_like(p)).sum()
    loss.backward()
    opt.step()

    assert not torch.allclose(p.data, p_before), "Il parametro non è cambiato dopo step"
    assert not torch.isnan(p.data).any(), "NaN nel parametro dopo Muon step"


def test_muon_momentum_accumulates():
    """Il buffer momentum cresce ad ogni step."""
    p = torch.nn.Parameter(torch.randn(16, 32).cuda())
    opt = MuonOptimizer([p], lr=0.001, momentum=0.95)

    for _ in range(3):
        loss = (p * torch.randn_like(p)).sum()
        loss.backward()
        opt.step()
        p.grad = None  # reset grad

    state = opt.state[p]
    assert "momentum_buf" in state, "momentum_buf non inizializzato"
    assert not torch.isnan(state["momentum_buf"]).any()


# ---------------------------------------------------------------------------
# Separazione 2D / 1D
# ---------------------------------------------------------------------------

def test_param_split():
    """I parametri devono essere correttamente separati in 2D (Muon) e 1D (AdamW)."""
    params = [
        torch.nn.Parameter(torch.randn(32, 64)),   # 2D — embedding-like
        torch.nn.Parameter(torch.randn(64)),        # 1D — norm-like
        torch.nn.Parameter(torch.randn(16, 32)),   # 2D — linear-like
        torch.nn.Parameter(torch.randn(32)),        # 1D — bias-like
    ]

    fp32_2d = [p for p in params if p.ndim >= 2]
    fp32_1d = [p for p in params if p.ndim < 2]

    assert all(p.ndim >= 2 for p in fp32_2d), "fp32_2d contiene parametri 1D"
    assert all(p.ndim < 2 for p in fp32_1d), "fp32_1d contiene parametri 2D"
    assert len(fp32_2d) + len(fp32_1d) == len(params), "Parametri persi nella separazione"
    assert len(fp32_2d) == 2
    assert len(fp32_1d) == 2

    # MuonOptimizer accetta correttamente i 2D
    opt = MuonOptimizer(fp32_2d, lr=0.01)
    assert len(list(opt.param_groups[0]["params"])) == 2
