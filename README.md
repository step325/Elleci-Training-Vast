# Elleci Training

Hybrid LLM: Mamba-2 SSD + Differential Attention + EG-MLA + BitNet INT2.

## Quick Start (vast.ai / A100)

```bash
# 1. Clone
git clone https://github.com/step325/Elleci-Training-Vast.git && cd elleci-train

# 2. Setup (installs deps, compiles CUDA kernels, runs dry-run test)
chmod +x setup.sh && ./setup.sh

# 3. Train
nohup python3 -u train.py --config configs/a100_7b.yaml > log/training_7b.log 2>&1 &

# 4. Monitor
tail -f log/training_7b.log
```

## Configurations

| Config | GPU | d_model | layers | params | batch | seq_len |
|--------|-----|---------|--------|--------|-------|---------|
| `configs/a100_7b.yaml` | A100 40GB | 4096 | 32 | 7.3B | 8 | 2048 |
| `configs/rtx4070_s.yaml` | RTX 4070 12GB | 1536 | 20 | 0.72B | 2 | 512 |

## Training Options

```bash
python3 train.py --config configs/a100_7b.yaml          # Full training
python3 train.py --config configs/a100_7b.yaml --dry-run # Quick test (tiny model)
python3 train.py --config configs/a100_7b.yaml --no-wandb # Without W&B logging
python3 train.py --resume checkpoints/7b_training/step_5000.pt  # Resume
```

## VRAM Budget (A100 7B)

| Component | GB |
|-----------|-----|
| INT2 weights (7B x 0.25B/param) | 1.75 |
| Hysteresis counters | 3.50 |
| FP32 params (embedding/norms) | 1.20 |
| Activations (batch=8, seq=2048) | 14.5 |
| Optimizer (AdamW for FP32) | 3.6 |
| Temporary | 2.0 |
| **Total** | **~26.5 / 40** |

