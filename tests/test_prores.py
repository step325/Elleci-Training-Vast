"""
Task 2 — ProRes (Progressive Residual Warmup)
Valida: residual_alpha buffer, set_residual_alpha, schedule warmup per layer.
"""
import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockBlock:
    """Blocco minimale con residual_alpha e set_residual_alpha (replica da conftest)."""

    def __init__(self):
        self.residual_alpha = torch.tensor(1.0)

    def set_residual_alpha(self, alpha: float):
        self.residual_alpha.fill_(alpha)


# ---------------------------------------------------------------------------
# Buffer residual_alpha su MockBlock
# ---------------------------------------------------------------------------

def test_alpha_init():
    """residual_alpha deve essere inizializzato a 1.0."""
    block = MockBlock()
    assert block.residual_alpha.item() == 1.0


def test_set_alpha():
    """set_residual_alpha modifica correttamente il buffer."""
    block = MockBlock()
    block.set_residual_alpha(0.3)
    assert abs(block.residual_alpha.item() - 0.3) < 1e-6

    block.set_residual_alpha(0.0)
    assert block.residual_alpha.item() == 0.0

    block.set_residual_alpha(1.0)
    assert block.residual_alpha.item() == 1.0


# ---------------------------------------------------------------------------
# Schedule warmup via mock trainer
# ---------------------------------------------------------------------------

def test_alpha_zero(prores_trainer):
    """Al step 0 tutti i blocchi hanno alpha = 0."""
    prores_trainer._update_prores_alpha(0)
    for i, block in enumerate(prores_trainer.model.blocks):
        alpha = block.residual_alpha.item()
        assert alpha < 0.05, f"Block {i}: alpha={alpha:.4f} dovrebbe essere ~0 al step 0"


def test_alpha_depth(prores_trainer):
    """A metà warmup: layer superficiale ha alpha > layer profondo."""
    prores_trainer._update_prores_alpha(50)  # metà del warmup=100

    alpha_first = prores_trainer.model.blocks[0].residual_alpha.item()
    alpha_last  = prores_trainer.model.blocks[-1].residual_alpha.item()

    assert alpha_first > alpha_last, (
        f"Layer 0 (alpha={alpha_first:.3f}) dovrebbe avere alpha > "
        f"layer profondo (alpha={alpha_last:.3f}) durante warmup"
    )
    # alpha_first deve essere > 0 (non ancora a 1.0)
    assert 0.0 < alpha_first <= 1.0


def test_alpha_warmup(prores_trainer):
    """Dopo warmup (step 2x warmup) tutti i blocchi raggiungono alpha=1.0."""
    prores_trainer._update_prores_alpha(200)  # > warmup * max_factor

    for i, block in enumerate(prores_trainer.model.blocks):
        alpha = block.residual_alpha.item()
        assert alpha == 1.0, f"Block {i}: alpha={alpha:.4f} dovrebbe essere 1.0 dopo warmup"


def test_alpha_monotone(prores_trainer):
    """Alpha del block 0 è non-decrescente al crescere dello step."""
    alphas = []
    for step in range(0, 150, 10):
        prores_trainer._update_prores_alpha(step)
        alphas.append(prores_trainer.model.blocks[0].residual_alpha.item())

    for i in range(len(alphas) - 1):
        assert alphas[i] <= alphas[i + 1], (
            f"Alpha non monotona: step {i*10}→{(i+1)*10}: {alphas[i]:.3f}→{alphas[i+1]:.3f}"
        )


# ---------------------------------------------------------------------------
# Correttezza matematica
# ---------------------------------------------------------------------------

def test_residual_scaling_no_nan():
    """alpha * tensor non produce NaN per qualsiasi alpha in [0, 1]."""
    block = MockBlock()
    x = torch.randn(2, 32, 128)

    for alpha in [0.0, 0.01, 0.1, 0.5, 0.99, 1.0]:
        block.set_residual_alpha(alpha)
        residual = block.residual_alpha * x
        assert not torch.isnan(residual).any(), f"NaN con alpha={alpha}"
        assert not torch.isinf(residual).any(), f"Inf con alpha={alpha}"
