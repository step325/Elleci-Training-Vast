#!/usr/bin/env bash
# deploy_optimizations.sh — Carica ottimizzazioni su server Vast.ai e avvia training
# Uso: ./deploy_optimizations.sh <host> <porta>
# Es:  ./deploy_optimizations.sh 50.173.192.62 41441

set -euo pipefail

HOST="${1:-50.173.192.62}"
PORT="${2:-41441}"
KEY="$HOME/.ssh/elleci_vastai"
REMOTE="root@${HOST}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

SSH="ssh -p ${PORT} -i ${KEY} -o StrictHostKeyChecking=no"
SCP="scp -P ${PORT} -i ${KEY} -o StrictHostKeyChecking=no"

echo "=== Deploy ottimizzazioni Elleci 7B ==="
echo "Server: ${HOST}:${PORT}"

# 1. Test connessione
echo ""
echo "[1/4] Test connessione..."
$SSH ${REMOTE} 'echo "OK: $(hostname)"'

# 2. Copia file modificati
echo ""
echo "[2/4] Copia file modificati..."

$SCP "${LOCAL_DIR}/train.py" \
     "${REMOTE}:/root/elleci-train/train.py"
echo "  ✓ train.py"

$SCP "${LOCAL_DIR}/model/model.py" \
     "${REMOTE}:/root/elleci-train/model/model.py"
echo "  ✓ model/model.py"

$SCP "${LOCAL_DIR}/core/bitnet/int2_linear_tc.py" \
     "${REMOTE}:/root/elleci-train/core/bitnet/int2_linear_tc.py"
echo "  ✓ core/bitnet/int2_linear_tc.py"

$SCP "${LOCAL_DIR}/core/bitnet/int2_linear_tc_offload.py" \
     "${REMOTE}:/root/elleci-train/core/bitnet/int2_linear_tc_offload.py"
echo "  ✓ core/bitnet/int2_linear_tc_offload.py"

$SCP "${LOCAL_DIR}/core/bitnet/bitnet_int2_optimized.py" \
     "${REMOTE}:/root/elleci-train/core/bitnet/bitnet_int2_optimized.py"
echo "  ✓ core/bitnet/bitnet_int2_optimized.py"

$SCP "${LOCAL_DIR}/configs/a100_7b.yaml" \
     "${REMOTE}:/root/elleci-train/configs/a100_7b.yaml"
echo "  ✓ configs/a100_7b.yaml"

$SCP "${LOCAL_DIR}/OPTIMIZATION_ROADMAP.md" \
     "${REMOTE}:/root/elleci-train/OPTIMIZATION_ROADMAP.md"
echo "  ✓ OPTIMIZATION_ROADMAP.md"

# 3. Test rapido (dry-run 5 step, config diagnosi)
echo ""
echo "[3/4] Test rapido sul server (dry-run)..."
$SSH ${REMOTE} \
  'cd /root/elleci-train && python3 train.py --dry-run --no-wandb 2>&1 | head -80'

# 4. Ferma training corrente e riavvia
echo ""
echo "[4/4] Riavvio training..."
$SSH ${REMOTE} '
  tmux kill-session -t training 2>/dev/null && echo "  Stopped old session" || echo "  No session to stop"
  sleep 1
  tmux new-session -d -s training \
    "cd /root/elleci-train && PYTHONUNBUFFERED=1 python3 train.py \
     --config configs/a100_7b.yaml --no-wandb 2>&1 | tee /tmp/training.log"
  echo "  Training avviato in tmux session: training"
  sleep 3
  echo ""
  echo "=== Prime righe del log ==="
  head -50 /tmp/training.log 2>/dev/null || echo "(log non ancora disponibile)"
'

echo ""
echo "=== Deploy completato ==="
echo "Per monitorare: ssh -p ${PORT} -i ${KEY} ${REMOTE} 'tmux attach -t training'"
echo "Per log:        ssh -p ${PORT} -i ${KEY} ${REMOTE} 'tail -f /tmp/training.log'"
