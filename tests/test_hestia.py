"""
Task 3 — HESTIA (Threshold Annealing)
Valida: _compute_hestia_threshold — clamp al range INT4 signed.
"""
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Valori ai punti chiave
# ---------------------------------------------------------------------------

def test_threshold_init(hestia_trainer):
    """Al step 0 il threshold viene clamped al massimo rappresentabile da INT4."""
    t = hestia_trainer._compute_hestia_threshold(0)
    assert t == 7, f"threshold(0)={t}, atteso 7"


def test_threshold_final(hestia_trainer):
    """Al warmup_steps e oltre il threshold deve essere il valore finale (7)."""
    assert hestia_trainer._compute_hestia_threshold(2000) == 7
    assert hestia_trainer._compute_hestia_threshold(3000) == 7
    assert hestia_trainer._compute_hestia_threshold(50000) == 7


def test_threshold_mid(hestia_trainer):
    """A metà warmup il threshold resta entro il massimo rappresentabile da INT4."""
    mid = hestia_trainer._compute_hestia_threshold(1000)
    assert mid == 7, f"threshold(1000)={mid}, atteso 7"


def test_threshold_monotone(hestia_trainer):
    """Il threshold deve essere non-crescente al crescere dello step."""
    steps = list(range(0, 2001, 100))
    thresholds = [hestia_trainer._compute_hestia_threshold(s) for s in steps]

    for i in range(len(thresholds) - 1):
        assert thresholds[i] >= thresholds[i + 1], (
            f"Threshold non monotono: step {steps[i]}→{steps[i+1]}: "
            f"{thresholds[i]}→{thresholds[i+1]}"
        )


def test_threshold_range(hestia_trainer):
    """Tutti i threshold devono essere rappresentabili nel formato INT4 signed."""
    for step in range(0, 2001, 50):
        t = hestia_trainer._compute_hestia_threshold(step)
        assert 0 <= t <= 7, f"threshold({step})={t} fuori dal range INT4 [0, 7]"


# ---------------------------------------------------------------------------
# Fallback senza int2_threshold_init
# ---------------------------------------------------------------------------

def test_threshold_fallback(hestia_trainer_no_init):
    """Senza int2_threshold_init il threshold è sempre uguale a int2_threshold (7)."""
    for step in [0, 100, 1000, 2000, 5000]:
        t = hestia_trainer_no_init._compute_hestia_threshold(step)
        assert t == 7, f"threshold({step})={t}, atteso 7 (fallback senza init)"
