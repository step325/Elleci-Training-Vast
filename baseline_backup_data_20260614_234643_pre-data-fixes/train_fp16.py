#!/usr/bin/env python3
"""
Elleci 3B - FP16/BF16 Standard Training
========================================

Standard pretraining WITHOUT INT2 quantization.
- AdamW for ALL parameters (BitLinear extends nn.Linear → fully compatible)
- Cosine LR schedule with warmup
- Gradient accumulation + clipping
- EllediDatasetV2: 3-phase curriculum (EN foundation → IT knowledge)
- Mamba checkpointing: memory-efficient via DifferentialMamba2BlockCheckpoint
- WandB logging (optional)
- Resume from checkpoint

Usage:
    python train_fp16.py --config configs/a100_3b_fp16.yaml
    python train_fp16.py --config configs/a100_3b_fp16.yaml --dry-run
    python train_fp16.py --config configs/a100_3b_fp16.yaml --resume checkpoints/fp16_3b/step_5000.pt
    python train_fp16.py --config configs/a100_3b_fp16.yaml --no-wandb

NOTE: After training, run scripts/quantize_model.py to convert to INT2 for inference.
"""
import os
import sys
import math
import time
import argparse
import yaml
from pathlib import Path

# Load .env file from script directory (WANDB_API_KEY, HF_TOKEN, etc.)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from model.config import (
    ElleciConfig, BitNetConfig, MLAConfig, MambaConfig,
    RouterConfig, ThinkingLoopConfig, MoEConfig,
)
from model.model import Elleci

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_elleci_config(config: dict, dry_run: bool = False) -> ElleciConfig:
    """Build ElleciConfig from YAML dict. Mirrors train.py's create_small_elleci_config."""
    model_cfg = config["model"]
    mamba_cfg = config.get("mamba", {})

    if dry_run:
        d_model = 256
        n_layers = 4
        vocab_size = model_cfg.get("vocab_size", 32128)
        max_seq_len = 128
        use_moe = False
    else:
        d_model = model_cfg["d_model"]
        n_layers = model_cfg["n_layers"]
        vocab_size = model_cfg.get("vocab_size", 32128)
        max_seq_len = model_cfg.get("max_seq_len", 2048)
        use_moe = model_cfg.get("use_moe", False)

    # Auto-detect n_heads for MLA
    n_heads_mla = model_cfg.get("n_heads", None)
    if n_heads_mla is None:
        if d_model >= 2048:
            n_heads_mla = 32
        elif d_model >= 1024:
            n_heads_mla = 16
        elif d_model >= 768:
            n_heads_mla = 12
        else:
            for n in [8, 4, 2, 1]:
                if d_model % n == 0:
                    n_heads_mla = n
                    break

    n_heads_mamba = mamba_cfg.get("n_heads", n_heads_mla)

    mla_config = MLAConfig(
        d_model=d_model,
        n_heads=n_heads_mla,
        kv_lora_rank=min(256, d_model // 4),
    )

    mamba_config = MambaConfig(
        d_model=d_model,
        d_state=mamba_cfg.get("d_state", 16),
        d_conv=4,
        expand=mamba_cfg.get("expand", 2),
        n_heads=n_heads_mamba,
        chunk_size=mamba_cfg.get("chunk_size", 32),
        use_mamba2=True,
        use_matmul_ssd=mamba_cfg.get("use_matmul_ssd", True),
        use_checkpointing=mamba_cfg.get("use_checkpointing", True),
        use_cuda_ssd=mamba_cfg.get("use_cuda_ssd", False),
    )

    moe_config = MoEConfig(
        d_model=d_model,
        num_experts=4,
        top_k=1,
        moe_layers=tuple(i for i in range(n_layers) if i % 4 != 3),
    )

    return ElleciConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        bitnet=BitNetConfig(),
        mla=mla_config,
        mamba=mamba_config,
        router=RouterConfig(d_model=d_model),
        thinking=ThinkingLoopConfig(),
        moe=moe_config,
        use_router=False,
        use_moe=use_moe,
        use_v2=True,
        dropout=0.0,   # No dropout for pretraining
    )


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: dict) -> float:
    """Cosine decay with linear warmup."""
    train_cfg = cfg["training"]
    warmup = train_cfg["warmup_steps"]
    max_steps = train_cfg["max_steps"]
    lr = train_cfg["lr"]
    min_lr = train_cfg["min_lr"]

    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(progress, 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW with weight decay only on 2D parameters."""
    train_cfg = cfg["training"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": train_cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=train_cfg["lr"],
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.95)),
        eps=train_cfg.get("eps", 1e-8),
    )


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, step, loss, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": cfg,
    }, path)

    limit = cfg.get("checkpointing", {}).get("save_total_limit", 5)
    ckpts = sorted(
        Path(output_dir).glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    while len(ckpts) > limit:
        ckpts.pop(0).unlink()

    print(f"  Checkpoint saved: {path}")
    return path


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["step"], ckpt.get("loss", float("inf"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SyntheticDataLoader:
    """Random token batches for dry-run / fallback."""
    def __init__(self, vocab_size, batch_size, seq_len, num_batches=1000):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
            yield {"input_ids": ids, "labels": ids.clone()}

    def __len__(self):
        return self.num_batches


def _get_phase(step: int, cfg: dict) -> int:
    """Return training phase (1, 2) based on current step."""
    train_cfg = cfg["training"]
    phase1_end = train_cfg.get("phase1_end_step", 35000)
    if step < phase1_end:
        return 1
    return 2


def build_dataloader(phase: int, cfg: dict, elleci_config: ElleciConfig):
    """Load EllediDatasetV2 for given phase. Falls back to synthetic on error."""
    train_cfg = cfg["training"]
    data_cfg = cfg.get("data", {})

    try:
        from data.elleci_dataset_v2 import EllediDatasetV2
        from tokenizers import Tokenizer

        tokenizer_path = ROOT_DIR / data_cfg.get("tokenizer_path", "tokenizer/tokenizer.json")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        raw_tok = Tokenizer.from_file(str(tokenizer_path))

        class _Wrap:
            def __init__(self, t):
                self._t = t
                self.eos_token_id = 2
                self.pad_token_id = 0
            def encode(self, text):
                return self._t.encode(text).ids

        dataset = EllediDatasetV2(
            tokenizer=_Wrap(raw_tok),
            phase=phase,
            max_length=train_cfg["seq_len"],
            batch_size=train_cfg["batch_size"],
            seed=data_cfg.get("seed", 42),
        )

        class _Loader:
            def __init__(self, ds):
                self._ds = ds
            def __iter__(self):
                for batch in self._ds:
                    yield {"input_ids": batch, "labels": batch.clone()}

        print(f"  Data: EllediDatasetV2 Phase {phase}")
        return _Loader(dataset)

    except Exception as e:
        print(f"  WARNING: Real data failed ({e}), using synthetic fallback")
        return SyntheticDataLoader(
            elleci_config.vocab_size,
            train_cfg["batch_size"],
            train_cfg["seq_len"],
            num_batches=cfg["training"]["max_steps"] * 4,
        )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, amp_dtype, max_batches=50):
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast(device_type="cuda" if device == "cuda" else "cpu",
                      dtype=amp_dtype,
                      enabled=(device == "cuda")):
            _, loss = model(ids, targets=labels)
        if loss is not None:
            total_loss += loss.item()
            count += 1
    model.train()
    avg_loss = total_loss / max(count, 1)
    return {"val_loss": avg_loss, "val_ppl": math.exp(min(avg_loss, 20))}


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(cfg: dict, dry_run: bool = False, resume_path: str = None,
          no_wandb: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cfg = cfg["training"]
    precision = train_cfg.get("mixed_precision", "bf16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Model ----
    print("\nBuilding model...")
    elleci_config = build_elleci_config(cfg, dry_run)
    model = Elleci(elleci_config).to(device)
    # FP16 training: convert model to bf16 on GPU
    if device == "cuda":
        model = model.to(amp_dtype)

    model.gradient_checkpointing_enable()  # 32 syncs layer-level vs 1536 chunk-level

    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,} ({total/1e9:.2f}B)")
    print(f"  Mamba chunk checkpointing: {elleci_config.mamba.use_checkpointing}")
    print(f"  Layer gradient checkpointing: True (model.gradient_checkpointing_enable)")
    print(f"  Precision: {precision}")

    if dry_run:
        print("\n[DRY RUN] Testing forward pass...")
        seq_len = min(train_cfg.get("seq_len", 2048), 128)
        batch_sz = min(train_cfg.get("batch_size", 4), 2)
        x = torch.randint(0, elleci_config.vocab_size, (batch_sz, seq_len), device=device)
        with torch.no_grad():
            with autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=amp_dtype,
                          enabled=(device == "cuda")):
                logits, loss = model(x, targets=x)
        print(f"  Input:  {x.shape}")
        print(f"  Output: {logits.shape}")
        print(f"  Loss:   {loss.item():.4f}")
        if device == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM used: {mem:.2f} GB")
            scale = train_cfg.get("seq_len", 2048) / seq_len
            print(f"  VRAM (est. full seq_len): ~{mem * scale:.1f} GB")
        print("\n[DRY RUN] Success!")
        return

    # ---- Optimizer ----
    optimizer = build_optimizer(model, cfg)

    # GradScaler only for fp16 (bf16 is numerically stable without it)
    scaler = GradScaler(enabled=(precision == "fp16"))

    # ---- Resume ----
    start_step = 0
    if resume_path:
        print(f"\nResuming from: {resume_path}")
        start_step, last_loss = load_checkpoint(resume_path, model, optimizer, device)
        print(f"  Resumed at step {start_step}, loss={last_loss:.4f}")

    # ---- Data ----
    print("\nPreparing data...")
    current_phase = _get_phase(start_step, cfg)
    train_loader = build_dataloader(current_phase, cfg, elleci_config)
    val_loader = SyntheticDataLoader(
        elleci_config.vocab_size,
        train_cfg["batch_size"],
        train_cfg["seq_len"],
        num_batches=200,
    )

    # ---- WandB ----
    log_cfg = cfg.get("logging", {})
    use_wandb = HAS_WANDB and not no_wandb
    if use_wandb:
        try:
            wandb.init(
                project=log_cfg.get("wandb_project", "elleci-fp16"),
                name=log_cfg.get("wandb_run_name", "3b-fp16"),
                config=cfg,
                resume="allow",
            )
        except Exception as e:
            print(f"  WandB init failed: {e}")
            use_wandb = False

    # ---- Training loop ----
    max_steps = train_cfg["max_steps"]
    accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
    log_every = log_cfg.get("log_interval", 50)
    eval_every = log_cfg.get("eval_interval", 1000)
    save_every = log_cfg.get("save_interval", 5000)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    output_dir = cfg.get("checkpointing", {}).get("output_dir", "checkpoints/fp16_3b")
    tokens_per_step = (
        train_cfg["batch_size"] * train_cfg["seq_len"] * accum_steps
    )

    print(f"\nStarting training: steps {start_step} → {max_steps}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {max_steps * tokens_per_step / 1e9:.2f}B")
    print(f"  Phase: {current_phase}")
    print()

    model.train()
    optimizer.zero_grad()

    step = start_step
    accum_loss = 0.0
    t_start = time.time()
    tokens_since_log = 0
    best_val_loss = float("inf")
    data_iter = iter(train_loader)

    while step < max_steps:
        # Phase switch
        new_phase = _get_phase(step, cfg)
        if new_phase != current_phase:
            current_phase = new_phase
            print(f"\n>>> Switching to Phase {current_phase} at step {step}")
            train_loader = build_dataloader(current_phase, cfg, elleci_config)
            data_iter = iter(train_loader)

        # LR
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Accumulation loop
        _micro_losses = []
        for _ in range(accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type="cuda" if device == "cuda" else "cpu",
                          dtype=amp_dtype,
                          enabled=(device == "cuda")):
                _, loss = model(ids, targets=labels)
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            _micro_losses.append(loss.detach())
            tokens_since_log += ids.numel()
        accum_loss += torch.stack(_micro_losses).sum().item()

        # Grad clip + optimizer step
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        step_loss = accum_loss
        accum_loss = 0.0

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - t_start
            tok_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
            ppl = math.exp(min(step_loss, 20))
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0

            print(
                f"step {step:>6d}/{max_steps} | "
                f"loss {step_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | gnorm {grad_norm:.2f} | "
                f"tok/s {tok_per_sec:.0f} | "
                f"mem {mem_gb:.1f}GB | phase {current_phase}"
            )

            if use_wandb:
                wandb.log({
                    "train/loss": step_loss,
                    "train/ppl": ppl,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mem_gb": mem_gb,
                    "train/phase": current_phase,
                    "step": step,
                })

            t_start = time.time()
            tokens_since_log = 0

        # Evaluation
        if step % eval_every == 0:
            eval_cfg = cfg.get("evaluation", {})
            metrics = evaluate(
                model, val_loader, device, amp_dtype,
                max_batches=eval_cfg.get("eval_samples", 200),
            )
            print(f"  [eval] loss={metrics['val_loss']:.4f} ppl={metrics['val_ppl']:.1f}")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(model, optimizer, step, metrics["val_loss"], cfg,
                                output_dir)
                print(f"  [eval] New best: {best_val_loss:.4f}")

            if use_wandb:
                wandb.log({
                    "eval/loss": metrics["val_loss"],
                    "eval/ppl": metrics["val_ppl"],
                    "step": step,
                })

        # Periodic checkpoint
        if step % save_every == 0:
            save_checkpoint(model, optimizer, step, step_loss, cfg, output_dir)

    # Final checkpoint
    save_checkpoint(model, optimizer, step, step_loss, cfg, output_dir)
    total_tokens = step * tokens_per_step
    print(f"\nTraining complete!")
    print(f"  Steps: {step}")
    print(f"  Tokens: {total_tokens / 1e9:.2f}B")
    print(f"  Best val loss: {best_val_loss:.4f}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Elleci FP16 Training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick forward pass test (no data loading)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable WandB logging")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # CLI resume overrides config
    if args.resume:
        cfg.setdefault("checkpointing", {})["resume_from"] = args.resume
    resume_path = cfg.get("checkpointing", {}).get("resume_from")

    train(cfg, dry_run=args.dry_run, resume_path=resume_path,
          no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
