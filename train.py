#!/usr/bin/env python3
"""
Training Elleci con INT2 - 50K Steps
=====================================

Questo script:
1. Carica Elleci e converte a INT2
2. DISABILITA: DeepSpeed, ZeRO, CPU Offload, Gradient Checkpointing
3. MANTIENE: Liger, Flash Attention, Mixed Precision
4. Esegue training 50K steps con hysteresis update

NOTA: Modello ridotto per RTX 4070 12GB (target <9GB VRAM)
"""

import os
import sys
import time
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Aggiungi paths
ROOT_DIR = Path(__file__).parent  # repo root (elleci-train/)
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "core", "bitnet"))
sys.path.insert(0, os.path.join(base_dir, "core", "mamba"))

# Import progetto localmente
from model.config import ElleciConfig
from model.model import Elleci

# VERSIONE OTTIMIZZATA: Usa Tensor Cores + INT8 + CPU Offload
try:
    from core.bitnet.bitnet_int2_optimized import (
        convert_bitnetlinear_to_int2,
        convert_all_linear_to_int2,
        get_int2_layers,
        get_non_int2_params,
        BitLinearInt2,
        prefetch_all_activations,
        setup_offload_manager,
        enable_hysteresis,
        disable_hysteresis,
        USING_OPTIMIZED
    )
    print("✓ Using OPTIMIZED INT2 (Tensor Cores + INT8 + CPU Offload)")
except ImportError as e:
    print(f"FATAL: Optimized INT2 module not available: {e}")
    print("  Make sure you're running from the repo root directory.")
    sys.exit(1)

# Liger kernels (opzionale)
try:
    from liger_kernel.transformers import LigerCrossEntropyLoss
    HAS_LIGER_CE = True
except ImportError:
    HAS_LIGER_CE = False
    print("Warning: Liger Cross-Entropy not available")

# Wandb (opzionale)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train Elleci INT2")
    parser.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "a100_7b.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Quick test with tiny model")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--local-rank", type=int, default=0)  # Per compatibilita
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Carica configurazione da YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class SyntheticDataLoader:
    """DataLoader sintetico per test."""

    def __init__(self, vocab_size, batch_size, seq_len, num_batches=10000):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.current = 0

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current >= self.num_batches:
            raise StopIteration
        self.current += 1

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def __len__(self):
        return self.num_batches


class INT2Trainer:
    """
    Trainer specializzato per INT2.

    Gestisce:
    - Hysteresis update per layer INT2
    - AdamW per embedding/norms
    - Mixed precision
    - Logging e checkpointing
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device

        # Estrai config
        train_cfg = config["training"]
        opt_cfg = config["optimizations"]

        # INT2 layers e parametri
        self.int2_layers = get_int2_layers(model)
        self.non_int2_params = get_non_int2_params(model)

        print(f"INT2 layers: {len(self.int2_layers)}")
        print(f"Non-INT2 params: {sum(p.numel() for p in self.non_int2_params):,}")

        # Optimizer SOLO per non-INT2 params
        if self.non_int2_params:
            self.optimizer = torch.optim.AdamW(
                self.non_int2_params,
                lr=train_cfg["adamw_lr"],
                weight_decay=train_cfg["weight_decay"]
            )
        else:
            self.optimizer = None

        # Mixed precision
        self.use_amp = train_cfg["mixed_precision"] in ["fp16", "bf16"]
        self.amp_dtype = torch.bfloat16 if train_cfg["mixed_precision"] == "bf16" else torch.float16
        self.scaler = GradScaler('cuda') if train_cfg["mixed_precision"] == "fp16" else None

        # Loss function
        if opt_cfg.get("use_liger_cross_entropy", False) and HAS_LIGER_CE:
            self.loss_fn = LigerCrossEntropyLoss()
            print("Using Liger Cross-Entropy")
        else:
            self.loss_fn = None  # Useremo F.cross_entropy

        # Training state
        self.global_step = 0
        self.tokens_seen = 0
        self.best_val_loss = float("inf")

        # Hysteresis config
        model_cfg = config["model"]
        self.int2_lr = train_cfg["int2_lr"]
        self.int2_threshold = model_cfg["int2_threshold"]
        self.int2_lr_scale = model_cfg["int2_lr_scale"]
        self.int2_decay = model_cfg["int2_decay_rate"]

        # Gradient accumulation
        self.grad_accum_steps = train_cfg["gradient_accumulation_steps"]
        self.accum_count = 0

        # Warmup
        self.warmup_steps = train_cfg["warmup_steps"]

    def get_lr(self, base_lr: float) -> float:
        """Learning rate con linear warmup."""
        if self.global_step < self.warmup_steps:
            return base_lr * (self.global_step + 1) / self.warmup_steps
        return base_lr

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calcola loss (con Liger se disponibile)."""
        # Shift per next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        vocab_size = shift_logits.size(-1)

        if self.loss_fn is not None:
            # Liger Cross-Entropy
            loss = self.loss_fn(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        else:
            # Standard Cross-Entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return loss

    def train_step(self, batch: dict) -> dict:
        """Singolo step di training."""
        self.model.train()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Control hysteresis: only enable on the last micro-batch of accumulation
        # to prevent INT2 weight changes between micro-batches (Issue #4 fix)
        is_last_micro_batch = (self.accum_count + 1 >= self.grad_accum_steps)
        if is_last_micro_batch:
            enable_hysteresis()
        else:
            disable_hysteresis()

        # Forward con mixed precision
        with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            # Forward - i layer INT2 salvano automaticamente l'input
            output = self.model(input_ids)

            # Estrai logits
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # Loss
            loss = self.compute_loss(logits, labels)
            loss = loss / self.grad_accum_steps  # Scale per accumulation

        # Backward
        # Prefetch activations from CPU before backward (async overlap)
        if is_last_micro_batch:
            prefetch_all_activations(self.model)

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.accum_count += 1

        # Update ogni grad_accum_steps
        if self.accum_count >= self.grad_accum_steps:
            # Gradient clipping
            if self.config["training"]["max_grad_norm"] > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )

            # Update non-INT2 params con AdamW
            if self.optimizer is not None:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Hysteresis update per INT2 layers
            current_lr = self.get_lr(self.int2_lr)
            self._hysteresis_update_all(current_lr)

            self.global_step += 1
            self.accum_count = 0

        # Tracking
        self.tokens_seen += input_ids.numel()

        return {
            "loss": loss.item() * self.grad_accum_steps,  # Unscale
            "lr": self.get_lr(self.int2_lr),
            "tokens": self.tokens_seen
        }

    def _hysteresis_update_all(self, lr: float):
        """
        Esegue hysteresis update su tutti i layer INT2.

        NOTA: I layer INT2 con grad hook aggiornano automaticamente
        durante il backward pass. _step is already incremented inside
        hysteresis_update(), so we must NOT increment it again here
        (that was causing the double-increment bug).
        """
        for layer in self.int2_layers:
            if hasattr(layer, 'int2_layer'):
                int2 = layer.int2_layer
            else:
                int2 = layer

            # Imposta LR per il prossimo backward
            if hasattr(int2, 'set_lr'):
                int2.set_lr(lr)

            # Refresh gamma cache (1 sync per layer, once per optimizer step instead of every fwd/bwd)
            if hasattr(int2, '_gamma_cached'):
                int2._gamma_cached = int2.gamma.detach().item()

            # NOTE: Do NOT increment _step here - it's already incremented
            # inside hysteresis_update() during backward pass.

    @torch.no_grad()
    def evaluate(self, dataloader, max_batches: int = 100) -> dict:
        """Valutazione su validation set."""
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(input_ids)

                if hasattr(output, "logits"):
                    logits = output.logits
                elif isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                loss = self.compute_loss(logits, labels)

            batch_tokens = input_ids.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_tokens": total_tokens
        }

    def save_checkpoint(self, path: str, extra_state: dict = None):
        """Salva checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Raccogli stato INT2
        int2_state = {}
        for i, layer in enumerate(self.int2_layers):
            if hasattr(layer, 'int2_layer'):
                int2 = layer.int2_layer
            else:
                int2 = layer

            # Sync Python step counter to buffer for checkpoint compatibility
            if hasattr(int2, '_step_py'):
                int2._step.fill_(int2._step_py)

            int2_state[f"layer_{i}"] = {
                "W_packed": int2.W_packed.cpu(),
                "H_packed": int2.H_packed.cpu(),
                "gamma": int2.gamma.data.cpu(),
                "step": int2._step.cpu() if hasattr(int2._step, 'cpu') else int2._step
            }

        checkpoint = {
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "int2_state": int2_state,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config,
        }

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Carica checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        # Carica model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Carica optimizer
        if self.optimizer and checkpoint.get("optimizer_state_dict"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Sync _step_py from buffer for all INT2 layers
        for layer in self.int2_layers:
            int2 = layer.int2_layer if hasattr(layer, 'int2_layer') else layer
            if hasattr(int2, '_step_py') and hasattr(int2, '_step'):
                int2._step_py = int2._step.item()
            if hasattr(int2, '_gamma_cached'):
                int2._gamma_cached = int2.gamma.detach().item()

        print(f"Checkpoint loaded from step {self.global_step}")

    def get_memory_stats(self) -> dict:
        """Statistiche memoria GPU."""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9
        }


def create_small_elleci_config(config: dict, dry_run: bool = False) -> ElleciConfig:
    """Crea config Elleci da YAML (supporta RTX 4070 e A100)."""
    from model.config import (
        ElleciConfig, BitNetConfig, MLAConfig, MambaConfig,
        RouterConfig, ThinkingLoopConfig, MoEConfig
    )

    model_cfg = config["model"]
    mamba_cfg = config.get("mamba", {})

    if dry_run:
        # Modello tiny per test
        d_model = 256
        n_layers = 4
        vocab_size = model_cfg.get("vocab_size", 32128)
        max_seq_len = 128
        use_moe = False
    else:
        d_model = model_cfg.get("d_model", 1280)
        n_layers = model_cfg.get("n_layers", 16)
        vocab_size = model_cfg.get("vocab_size", 32128)
        max_seq_len = model_cfg.get("max_seq_len", 512)
        use_moe = model_cfg.get("use_moe", False)

    # n_heads from config or auto-detect
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

    # Mamba n_heads: from config or same as MLA
    n_heads_mamba = mamba_cfg.get("n_heads", n_heads_mla)

    # Mamba settings from YAML (with defaults for backward compat)
    use_matmul_ssd = mamba_cfg.get("use_matmul_ssd", True)
    use_checkpointing = mamba_cfg.get("use_checkpointing", True)
    use_cuda_ssd = mamba_cfg.get("use_cuda_ssd", False)
    chunk_size = mamba_cfg.get("chunk_size", 32)
    d_state = mamba_cfg.get("d_state", 16)
    expand = mamba_cfg.get("expand", 2)

    # Crea sub-configs
    mla_config = MLAConfig(
        d_model=d_model,
        n_heads=n_heads_mla,
        kv_lora_rank=min(256, d_model // 4),
    )

    mamba_config = MambaConfig(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=expand,
        n_heads=n_heads_mamba,
        chunk_size=chunk_size,
        use_mamba2=True,
        use_matmul_ssd=use_matmul_ssd,
        use_checkpointing=use_checkpointing,
        use_cuda_ssd=use_cuda_ssd,
    )

    router_config = RouterConfig(d_model=d_model)

    moe_config = MoEConfig(
        d_model=d_model,
        num_experts=4,
        top_k=1,
        moe_layers=tuple(i for i in range(n_layers) if i % 4 != 3),
    )

    # Crea config principale
    elleci_config = ElleciConfig(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        bitnet=BitNetConfig(),
        mla=mla_config,
        mamba=mamba_config,
        router=router_config,
        thinking=ThinkingLoopConfig(),
        moe=moe_config,
        use_router=False,
        use_moe=use_moe,
        use_v2=True,
        dropout=0.1,
    )

    return elleci_config


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    config = load_config(args.config)

    # Dry run override
    if args.dry_run:
        print("\n>>> DRY RUN MODE - Using tiny model")
        config["training"]["max_steps"] = 100
        config["training"]["batch_size"] = 2
        config["training"]["seq_len"] = 128
        config["logging"]["log_interval"] = 5
        config["logging"]["eval_interval"] = 50
        config["logging"]["save_interval"] = 50

    # Crea config Elleci
    elleci_config = create_small_elleci_config(config, args.dry_run)

    print(f"\nModel config: n_layers={elleci_config.n_layers}, d_model={elleci_config.d_model}")
    print(f"MoE: {elleci_config.use_moe}")
    print(f"Training config: max_steps={config['training']['max_steps']}, "
          f"batch_size={config['training']['batch_size']}")

    # Verifica che ottimizzazioni incompatibili siano disabilitate
    opt_cfg = config["optimizations"]
    assert not opt_cfg.get("use_deepspeed", False), "DeepSpeed deve essere disabilitato per INT2"
    assert not opt_cfg.get("use_zero", False), "ZeRO deve essere disabilitato per INT2"
    # NOTE: gradient_checkpointing in optimizations refers to PyTorch's generic checkpointing
    # Mamba2's own use_checkpointing is separate and compatible with INT2
    print("\n[OK] Ottimizzazioni incompatibili correttamente disabilitate")

    # Mamba settings now come from YAML config (mamba section)
    print(f"Mamba config: matmul_ssd={elleci_config.mamba.use_matmul_ssd}, "
          f"checkpointing={elleci_config.mamba.use_checkpointing}, "
          f"chunk_size={elleci_config.mamba.chunk_size}")

    # Reset offload state
    from core.bitnet.int2_linear_tc_offload import Int2LinearTCOffloadWithGradHook
    Int2LinearTCOffloadWithGradHook._layer_counter = 0
    Int2LinearTCOffloadWithGradHook._manager = None
    Int2LinearTCOffloadWithGradHook._hysteresis_enabled = True

    # Crea modello
    print("\nCreating model...")
    model = Elleci(elleci_config)

    # Phase 2 optimization: disable CPU offload (sync elimination saves more than offload)
    # setup_offload_manager(300)  # Disabled: offload adds 378 GPU↔CPU ops/step

    # Converti a INT2 (tutti i layer lineari, non solo BitNetLinear)
    print("Converting ALL linear layers to INT2...")
    model_cfg = config["model"]
    enable_offload = config.get("training", {}).get("enable_offload", False)
    print(f"CPU offload: {'enabled' if enable_offload else 'disabled'}")
    model = convert_all_linear_to_int2(
        model,
        threshold=model_cfg["int2_threshold"],
        lr_scale=model_cfg["int2_lr_scale"],
        decay_rate=model_cfg["int2_decay_rate"],
        min_size=256,
        inplace=True,
        verbose=True,
        enable_offload=enable_offload
    ).to(device)

    # Stats modello
    total_params = sum(p.numel() for p in model.parameters())
    int2_layers = get_int2_layers(model)
    int2_params = sum(l.in_features * l.out_features for l in int2_layers)

    print(f"\nModel statistics:")
    print(f"  Total params: {total_params:,}")
    print(f"  INT2 params: {int2_params:,} ({int2_params/total_params*100:.1f}%)")
    print(f"  INT2 layers: {len(int2_layers)}")

    # Memory dopo creazione modello
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Model memory: {mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    # DataLoader
    print("\nPreparing data...")
    train_cfg = config["training"]

    # Load real data from FineWeb-edu streaming
    if args.dry_run:
        # Dry run: synthetic data for speed
        train_loader = SyntheticDataLoader(
            elleci_config.vocab_size,
            train_cfg["batch_size"],
            train_cfg["seq_len"],
            num_batches=config["training"]["max_steps"] * 2
        )
        val_loader = SyntheticDataLoader(
            elleci_config.vocab_size,
            train_cfg["batch_size"],
            train_cfg["seq_len"],
            num_batches=100
        )
        print("Using SYNTHETIC data (dry-run mode)")
    else:
        # Real training: use EllediDatasetV2 with FineWeb-edu
        try:
            from data.elleci_dataset_v2 import EllediDatasetV2
            from tokenizers import Tokenizer
            
            # Load tokenizer
            tokenizer_path = ROOT_DIR / config["data"]["tokenizer_path"]
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
            
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            
            # Create wrapper for compatibility
            class TokenizerWrapper:
                def __init__(self, tokenizer):
                    self._tokenizer = tokenizer
                    self.eos_token_id = 2  # Standard EOS
                    self.pad_token_id = 0  # Standard PAD
                
                def encode(self, text):
                    return self._tokenizer.encode(text).ids
            
            wrapped_tokenizer = TokenizerWrapper(tokenizer)
            
            # Create training dataset (Phase 1: English Foundation)
            print("Loading FineWeb-edu streaming dataset...")
            train_dataset = EllediDatasetV2(
                tokenizer=wrapped_tokenizer,
                phase=1,  # FineWeb-edu + Cosmopedia + OpenWebMath + Stack
                max_length=train_cfg["seq_len"],
                batch_size=train_cfg["batch_size"],
                seed=42
            )
            
            # Validation uses synthetic to avoid duplicating streaming data
            val_loader = SyntheticDataLoader(
                elleci_config.vocab_size,
                train_cfg["batch_size"],
                train_cfg["seq_len"],
                num_batches=100
            )
            
            # Wrap dataset as iterator-style loader
            class RealDataLoader:
                def __init__(self, dataset, max_batches):
                    self.dataset = dataset
                    self.max_batches = max_batches
                    
                def __iter__(self):
                    count = 0
                    for batch in self.dataset:
                        if count >= self.max_batches:
                            break
                        yield {"input_ids": batch, "labels": batch.clone()}
                        count += 1
                        
                def __len__(self):
                    return self.max_batches
            
            train_loader = RealDataLoader(train_dataset, config["training"]["max_steps"] * 2)
            print("✓ Using REAL data (FineWeb-edu + Cosmopedia + OpenWebMath + Stack)")
            
        except Exception as e:
            print(f"⚠ Failed to load real data: {e}")
            print("  Falling back to synthetic data...")
            train_loader = SyntheticDataLoader(
                elleci_config.vocab_size,
                train_cfg["batch_size"],
                train_cfg["seq_len"],
                num_batches=config["training"]["max_steps"] * 2
            )
            val_loader = SyntheticDataLoader(
                elleci_config.vocab_size,
                train_cfg["batch_size"],
                train_cfg["seq_len"],
                num_batches=100
            )
            print("Using SYNTHETIC data (fallback)")

    # Trainer
    trainer = INT2Trainer(model, config, device)

    # Resume se specificato
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif config["checkpointing"].get("resume_from"):
        trainer.load_checkpoint(config["checkpointing"]["resume_from"])

    # Wandb
    if HAS_WANDB and not args.no_wandb and not args.dry_run:
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=config["logging"]["wandb_run_name"],
            config=config
        )

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config['training']['max_steps']} steps...")
    print(f"{'='*60}\n")

    log_cfg = config["logging"]
    ckpt_cfg = config["checkpointing"]
    max_steps = config["training"]["max_steps"]

    os.makedirs(ckpt_cfg["output_dir"], exist_ok=True)

    start_time = time.time()
    running_loss = 0.0
    log_steps = 0

    train_iter = iter(train_loader)

    while trainer.global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train step
        metrics = trainer.train_step(batch)

        if metrics["loss"] > 0:  # Solo dopo gradient accumulation
            running_loss += metrics["loss"]
            log_steps += 1

        # Logging (solo dopo gradient accumulation completa)
        if trainer.global_step > 0 and trainer.global_step % log_cfg["log_interval"] == 0 and trainer.accum_count == 0:
            avg_loss = running_loss / max(log_steps, 1)
            elapsed = time.time() - start_time
            tokens_per_sec = trainer.tokens_seen / elapsed
            mem_stats = trainer.get_memory_stats()

            ppl = torch.exp(torch.tensor(avg_loss)).item()
            if ppl > 100000:
                ppl_str = f"{ppl:.2e}"
            else:
                ppl_str = f"{ppl:.2f}"

            log_msg = (
                f"Step {trainer.global_step:6d} | "
                f"Loss: {avg_loss:.4f} | "
                f"PPL: {ppl_str} | "
                f"LR: {metrics['lr']:.4f} | "
                f"Tok/s: {tokens_per_sec:.0f}"
            )

            if mem_stats:
                log_msg += f" | Mem: {mem_stats['peak_gb']:.1f}GB"

            print(log_msg)

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/perplexity": ppl,
                    "train/lr": metrics["lr"],
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_seen": trainer.tokens_seen,
                    "memory/peak_gb": mem_stats.get("peak_gb", 0),
                }, step=trainer.global_step)

            running_loss = 0.0
            log_steps = 0

        # Evaluation (solo dopo gradient accumulation completa)
        if trainer.global_step > 0 and trainer.global_step % log_cfg["eval_interval"] == 0 and trainer.accum_count == 0:
            val_metrics = trainer.evaluate(val_loader, max_batches=50)

            print(f"  >> Eval: loss={val_metrics['val_loss']:.4f}, "
                  f"ppl={val_metrics['val_perplexity']:.2f}")

            if HAS_WANDB and wandb.run:
                wandb.log({
                    "eval/loss": val_metrics["val_loss"],
                    "eval/perplexity": val_metrics["val_perplexity"],
                }, step=trainer.global_step)

            # Best model
            if val_metrics["val_loss"] < trainer.best_val_loss:
                trainer.best_val_loss = val_metrics["val_loss"]
                trainer.save_checkpoint(
                    os.path.join(ckpt_cfg["output_dir"], "best.pt")
                )

        # Checkpoint (solo dopo gradient accumulation completa)
        if trainer.global_step > 0 and trainer.global_step % log_cfg["save_interval"] == 0 and trainer.accum_count == 0:
            trainer.save_checkpoint(
                os.path.join(ckpt_cfg["output_dir"], f"step_{trainer.global_step}.pt")
            )

    # Final save
    trainer.save_checkpoint(
        os.path.join(ckpt_cfg["output_dir"], "final.pt")
    )

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"  Total steps: {trainer.global_step}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Tokens seen: {trainer.tokens_seen:,}")
    print(f"  Best val loss: {trainer.best_val_loss:.4f}")
    print(f"  Best val PPL: {torch.exp(torch.tensor(trainer.best_val_loss)).item():.2f}")

    if HAS_WANDB and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
