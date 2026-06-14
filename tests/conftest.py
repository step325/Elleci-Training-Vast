"""
Fixture condivise per la suite di validazione ottimizzazioni Elleci.

Usa mock leggeri per evitare di costruire il trainer completo nei test unitari.
Il test di integrazione (validate.sh) usa il dry-run reale per coprire il modello intero.
"""
import os
import sys
import types
import pytest
import torch
from pathlib import Path

# Aggiunge la root del repo al path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

# Assicura che nvcc e ninja siano in PATH durante i test
venv_bin = str(REPO / ".venv" / "bin")
cuda_bin = "/opt/cuda/bin"
current_path = os.environ.get("PATH", "")
if venv_bin not in current_path:
    os.environ["PATH"] = f"{venv_bin}:{cuda_bin}:{current_path}"


# ---------------------------------------------------------------------------
# Mock per test HESTIA (_compute_hestia_threshold)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hestia_trainer():
    """Mock trainer con init fuori range per verificare il clamp INT4."""
    from train import INT2Trainer

    obj = types.SimpleNamespace()
    obj.config = {
        "model": {"int2_threshold": 7, "int2_threshold_init": 14},
    }
    obj.warmup_steps = 2000
    obj._compute_hestia_threshold = lambda step: INT2Trainer._compute_hestia_threshold(obj, step)
    return obj


@pytest.fixture(scope="module")
def hestia_trainer_no_init():
    """Mock trainer senza int2_threshold_init (fallback: nessun annealing)."""
    from train import INT2Trainer

    obj = types.SimpleNamespace()
    obj.config = {
        "model": {"int2_threshold": 7},
    }
    obj.warmup_steps = 2000
    obj._compute_hestia_threshold = lambda step: INT2Trainer._compute_hestia_threshold(obj, step)
    return obj


# ---------------------------------------------------------------------------
# Mock per test ProRes (_update_prores_alpha)
# ---------------------------------------------------------------------------

class MockBlock:
    """Blocco minimale con residual_alpha e set_residual_alpha."""

    def __init__(self):
        self.residual_alpha = torch.tensor(1.0)

    def set_residual_alpha(self, alpha: float):
        self.residual_alpha.fill_(alpha)


@pytest.fixture
def prores_trainer():
    """Mock trainer ProRes: warmup=100, 4 blocchi."""
    from train import INT2Trainer

    obj = types.SimpleNamespace()
    obj.warmup_steps = 100
    obj.model = types.SimpleNamespace(blocks=[MockBlock() for _ in range(4)])
    obj._update_prores_alpha = lambda step: INT2Trainer._update_prores_alpha(obj, step)
    return obj
