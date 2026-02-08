#!/bin/bash
# ============================================================
# Elleci - Environment Setup (vast.ai / A100)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  Elleci Training - Environment Setup"
echo "========================================"

# 1. Verify GPU
echo ""
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available!')
    exit(1)
major, minor = torch.cuda.get_device_capability()
print(f'CUDA capability: sm_{major}{minor}')
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'VRAM: {vram:.1f} GB')
if vram < 35:
    print('WARNING: Less than 35GB VRAM. Use configs/rtx4070_s.yaml instead.')
"

# 2. Install dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install --quiet torch>=2.5.0 transformers>=4.45.0 tokenizers>=0.20.0
pip install --quiet einops>=0.8.0 datasets>=3.0.0
pip install --quiet wandb>=0.18.0 tqdm>=4.66.0
pip install --quiet liger-kernel>=0.6.0
pip install --quiet pyyaml

# Optional
pip install --quiet causal-conv1d>=1.4.0 2>/dev/null || echo "  causal-conv1d skipped (optional)"
pip install --quiet mamba-ssm>=2.0.0 2>/dev/null || echo "  mamba-ssm skipped (using custom impl)"
pip install --quiet accelerated-scan>=0.2.0 2>/dev/null || echo "  accelerated-scan skipped (optional)"

# 3. Pre-compile CUDA kernels
echo ""
echo "[3/6] Pre-compiling CUDA kernels (2-5 minutes)..."
python3 -c "
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'core/bitnet')
print('Compiling INT2 Tensor Core kernels...')
from core.bitnet.int2_linear_tc import HAS_TC_OPS, _CUDA_ARCH
if HAS_TC_OPS:
    print(f'  INT2 kernels: OK ({_CUDA_ARCH})')
else:
    print('  INT2 kernels: FAILED')
    sys.exit(1)
print('Kernel compilation complete!')
"

# 4. Verify tokenizer
echo ""
echo "[4/6] Checking tokenizer..."
if [ -f "$SCRIPT_DIR/tokenizer/tokenizer.json" ]; then
    echo "  Tokenizer found."
else
    echo "  WARNING: Tokenizer not found at tokenizer/tokenizer.json"
    echo "  Training will use synthetic data unless you provide it."
fi

# 5. Create output directories
echo ""
echo "[5/6] Creating output directories..."
mkdir -p checkpoints/7b_training
mkdir -p log

# 6. Dry-run test
echo ""
echo "[6/6] Running dry-run test..."
python3 train.py --dry-run --no-wandb 2>&1 | tail -20

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start training:"
echo "  nohup python3 -u train.py --config configs/a100_7b.yaml > log/training_7b.log 2>&1 &"
echo ""
echo "To monitor:"
echo "  tail -f log/training_7b.log"
