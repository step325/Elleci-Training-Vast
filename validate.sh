#!/usr/bin/env bash
# validate.sh — Suite di validazione automatica ottimizzazioni Elleci 7B
#
# Uso: cd elleci-train && bash validate.sh
# Exit 0 = tutti i test passano; Exit 1 = almeno un test fallisce
#
# Test coperti:
#   Task 1: Muon Optimizer (ns_step 2D/3D/tall, MuonOptimizer, split 2D/1D)
#   Task 2: ProRes (residual_alpha init, set, schedule warmup, monotonia)
#   Task 3: HESTIA (threshold 14→7, monotonia, fallback)
#   Task 4: Sparse-BitNet (2 non-zero/gruppo, shape, valori, padding)
#   Task 5: BitNet v2 (WHT involutoria, packed shape, INT4 range, RMS < 5%)
#   Integrazione: dry-run 100 step — nessun crash, "Training Complete"

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="$(pwd)/.venv/bin/python"
export PATH="$(pwd)/.venv/bin:/opt/cuda/bin:${PATH}"

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

PASS=0
FAIL=0
FAILED_TESTS=()

# ---------------------------------------------------------------------------
run_test() {
    local name="$1"
    local cmd="$2"
    printf "  %-50s" "$name"
    if eval "$cmd" > /tmp/_elleci_test_out.txt 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++)) || true
    else
        echo -e "${RED}FAIL${NC}"
        ((FAIL++)) || true
        FAILED_TESTS+=("$name")
        # Mostra le ultime righe di errore indentate
        tail -15 /tmp/_elleci_test_out.txt | sed 's/^/    /'
        echo ""
    fi
}

run_pytest() {
    local name="$1"
    local test_id="$2"
    run_test "$name" "$PYTHON -m pytest tests/$test_id -q --tb=short 2>&1"
}

# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "   Elleci — Validazione Ottimizzazioni (RTX 4070)"
echo "========================================================"
echo ""

# Verifica dipendenze
printf "Controllo dipendenze..."
if ! $PYTHON -c "import torch, pytest" 2>/dev/null; then
    echo -e " ${YELLOW}installazione pytest...${NC}"
    uv pip install --python "$PYTHON" pytest -q
else
    echo -e " ${GREEN}OK${NC}"
fi
echo ""

# ---------------------------------------------------------------------------
echo "[1/6] Task 1 — Muon Optimizer"
run_pytest "ns_step su tensore 2D"              "test_muon.py::test_ns_step_2d"
run_pytest "ns_step su tensore 3D (conv Mamba)" "test_muon.py::test_ns_step_3d"
run_pytest "ns_step su tensore tall"            "test_muon.py::test_ns_step_tall"
run_pytest "ns_step su tensore quadrato (sv)"   "test_muon.py::test_ns_step_square"
run_pytest "MuonOptimizer step aggiorna param"  "test_muon.py::test_muon_step"
run_pytest "MuonOptimizer momentum accumula"    "test_muon.py::test_muon_momentum_accumulates"
run_pytest "Separazione params 2D/1D"           "test_muon.py::test_param_split"

echo ""
echo "[2/6] Task 2 — ProRes (Progressive Residual Warmup)"
run_pytest "residual_alpha init = 1.0"          "test_prores.py::test_alpha_init"
run_pytest "set_residual_alpha funziona"        "test_prores.py::test_set_alpha"
run_pytest "alpha = 0 al step 0"                "test_prores.py::test_alpha_zero"
run_pytest "layer profondo < layer superficiale" "test_prores.py::test_alpha_depth"
run_pytest "alpha = 1.0 dopo warmup"            "test_prores.py::test_alpha_warmup"
run_pytest "alpha monotona crescente"           "test_prores.py::test_alpha_monotone"
run_pytest "alpha * tensor: no NaN"             "test_prores.py::test_residual_scaling_no_nan"

echo ""
echo "[3/6] Task 3 — HESTIA (Threshold Annealing)"
run_pytest "threshold(0) == 14"                 "test_hestia.py::test_threshold_init"
run_pytest "threshold(warmup) == 7"             "test_hestia.py::test_threshold_final"
run_pytest "threshold(warmup/2) in [10,11]"     "test_hestia.py::test_threshold_mid"
run_pytest "threshold monotona decrescente"     "test_hestia.py::test_threshold_monotone"
run_pytest "threshold in range [7,14]"          "test_hestia.py::test_threshold_range"
run_pytest "fallback senza threshold_init"      "test_hestia.py::test_threshold_fallback"

echo ""
echo "[4/6] Task 4 — Sparse-BitNet (2:4 Sparsity)"
run_pytest "esattamente 2 non-zero per gruppo"  "test_sparse_bitnet.py::test_24_count"
run_pytest "2 non-zero su tensore random"       "test_sparse_bitnet.py::test_24_count_random"
run_pytest "valori rimasti in {-1,0,1}"         "test_sparse_bitnet.py::test_24_values"
run_pytest "shape preservata"                   "test_sparse_bitnet.py::test_24_shape"
run_pytest "K non multiplo di 4 (padding)"      "test_sparse_bitnet.py::test_24_padding"
run_pytest "top-2 per valore assoluto"          "test_sparse_bitnet.py::test_24_top2_selection"

echo ""
echo "[5/6] Task 5 — BitNet v2 (Hadamard + INT4)"
run_pytest "WHT involutoria H(H(x))=x"          "test_bitnet_v2.py::test_wht_involutory"
run_pytest "WHT preserva la shape"              "test_bitnet_v2.py::test_wht_shape_preserved"
run_pytest "WHT ortonormale (norma conservata)" "test_bitnet_v2.py::test_wht_orthogonal"
run_pytest "packed shape [M, K//2]"             "test_bitnet_v2.py::test_packed_shape"
run_pytest "packed shape batch 3D"              "test_bitnet_v2.py::test_packed_shape_batch"
run_pytest "INT4 valori in [-7, 7]"             "test_bitnet_v2.py::test_int4_range"
run_pytest "round-trip RMS error < 5%"          "test_bitnet_v2.py::test_roundtrip_rms"
run_pytest "round-trip no NaN/Inf"              "test_bitnet_v2.py::test_roundtrip_no_nan"
run_pytest "round-trip shape preservata"        "test_bitnet_v2.py::test_roundtrip_shape"

# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Integrazione — dry-run 100 step"
printf "  %-50s" "dry-run senza crash (Training Complete)"
DRY_LOG=/tmp/elleci_dryrun.log
if $PYTHON train.py --dry-run --no-wandb > "$DRY_LOG" 2>&1 \
   && grep -q "Training Complete" "$DRY_LOG"; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++)) || true
    # Mostra ultimi step del log
    grep "^Step\|Training Complete\|Best val" "$DRY_LOG" | tail -5 | sed 's/^/    /'
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++)) || true
    FAILED_TESTS+=("dry-run integrazione")
    tail -30 "$DRY_LOG" | sed 's/^/    /'
fi

printf "  %-50s" "nessun NaN/Inf/CUDA error nel log"
if ! grep -qE "NaN|Inf|CUDA error|RuntimeError|Traceback" "$DRY_LOG" 2>/dev/null; then
    echo -e "${GREEN}PASS${NC}"
    ((PASS++)) || true
else
    echo -e "${RED}FAIL${NC}"
    ((FAIL++)) || true
    FAILED_TESTS+=("dry-run: presenza NaN/errori")
    grep -E "NaN|Inf|CUDA error|RuntimeError|Traceback" "$DRY_LOG" | head -10 | sed 's/^/    /'
fi

# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
printf "  PASS: ${GREEN}%d${NC}  |  FAIL: " "$PASS"
if [ "$FAIL" -gt 0 ]; then
    printf "${RED}%d${NC}\n" "$FAIL"
else
    printf "${GREEN}%d${NC}\n" "$FAIL"
fi
echo "========================================================"

if [ "${#FAILED_TESTS[@]}" -gt 0 ]; then
    echo ""
    echo -e "${RED}Test falliti:${NC}"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}Tutti i test passati.${NC}"
exit 0
