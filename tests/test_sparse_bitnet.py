"""
Task 4 — Sparse-BitNet (2:4 Structured Sparsity)
Valida: apply_24_sparsity — esattamente 2 non-zero per gruppo di 4 elementi.
"""
import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.bitnet.int2_linear_tc import apply_24_sparsity


# ---------------------------------------------------------------------------
# Correttezza struttura 2:4
# ---------------------------------------------------------------------------

def test_24_count():
    """Ogni gruppo di 4 deve avere esattamente 2 elementi non-zero."""
    W = torch.tensor([[-1.0, 1.0, 0.0, -1.0,  0.0, 1.0, -1.0, 1.0]])  # shape (1, 8)
    W_sparse = apply_24_sparsity(W)

    W_grouped = W_sparse.view(-1, 4)  # (2, 4)
    nonzero_per_group = (W_grouped != 0).sum(dim=1)
    assert (nonzero_per_group == 2).all(), (
        f"Non-zero per gruppo: {nonzero_per_group.tolist()}, tutti devono essere 2"
    )


def test_24_count_random():
    """Verifica 2 non-zero per gruppo su input non-zero (tutti {-1,+1}, senza zeri)."""
    torch.manual_seed(42)
    # Usa {-1,+1} per garantire >=2 non-zero per gruppo → apply_24_sparsity deve dare esattamente 2
    W = (torch.randint(0, 2, (64, 128)).float() * 2 - 1)  # {-1, +1}, nessuno zero
    W_sparse = apply_24_sparsity(W)

    W_grouped = W_sparse.view(-1, 4)  # (64*128//4, 4) = (2048, 4)
    nonzero_per_group = (W_grouped != 0).sum(dim=1)
    assert (nonzero_per_group == 2).all(), (
        f"Gruppi con numero errato di non-zero: "
        f"{(nonzero_per_group != 2).sum().item()} su {len(nonzero_per_group)}"
    )


# ---------------------------------------------------------------------------
# Valori
# ---------------------------------------------------------------------------

def test_24_values():
    """I valori rimasti devono essere in {-1, 0, 1} (input ternario)."""
    torch.manual_seed(0)
    W = torch.randint(-1, 2, (32, 64)).float()
    W_sparse = apply_24_sparsity(W)

    unique_vals = set(W_sparse.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0}), (
        f"Valori non ternari trovati: {unique_vals - {-1.0, 0.0, 1.0}}"
    )


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_24_shape():
    """Shape deve essere preservata."""
    shapes = [(256, 512), (32, 128), (8, 8)]
    for shape in shapes:
        W = torch.randint(-1, 2, shape).float()
        W_sparse = apply_24_sparsity(W)
        assert W_sparse.shape == W.shape, f"Shape {shape}: {W_sparse.shape} != {shape}"


def test_24_padding():
    """Funziona anche con K non multiplo di 4 (padding automatico)."""
    W = torch.randint(-1, 2, (16, 30)).float()
    W_sparse = apply_24_sparsity(W)

    assert W_sparse.shape == (16, 30), f"Shape errata con K=30: {W_sparse.shape}"
    assert not torch.isnan(W_sparse).any()


# ---------------------------------------------------------------------------
# Top-2 corretto
# ---------------------------------------------------------------------------

def test_24_top2_selection():
    """Con input [0, -1, 1, -1] i due mantenuti devono essere quelli con |val|=1."""
    W = torch.tensor([[0.0, -1.0, 1.0, -1.0]])
    W_sparse = apply_24_sparsity(W)

    # Il valore 0 deve essere azzerato (|0|=0 è il minimo)
    assert W_sparse[0, 0].item() == 0.0, "Il valore 0 non è stato azzerato"
    # Esattamente 2 non-zero
    assert (W_sparse != 0).sum().item() == 2
