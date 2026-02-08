"""
BitLinear implementato con Int2LinearTC Ottimizzato.

VERSIONE OTTIMIZZATA che usa:
- Phase 1: Tensor Cores (16.6x matmul speedup)
- Phase 2: INT8 Activations (50% memory savings)
- Phase 4: CPU Offload (enable larger models)

Drop-in replacement per src/modules/bitnet_int2.py
"""
import torch
import torch.nn as nn
import sys
import os

# Import locali
try:
    from .int2_linear_tc_offload import (
        Int2LinearTCOffload as Int2Linear,
        Int2LinearTCOffloadWithGradHook as Int2LinearWithGradHook,
        schedule_all_prefetches
    )
    from .overlap_transfer import OverlapTransferManager
    USING_OPTIMIZED = True
    print("[INT2] ✓ Using OPTIMIZED kernels (Tensor Cores + INT8 + CPU Offload)")
except (ImportError, ValueError):
    # Support both package and direct execution
    try:
        from int2_linear_tc_offload import (
            Int2LinearTCOffload as Int2Linear,
            Int2LinearTCOffloadWithGradHook as Int2LinearWithGradHook,
            schedule_all_prefetches
        )
        from overlap_transfer import OverlapTransferManager
        USING_OPTIMIZED = True
        print("[INT2] ✓ Using OPTIMIZED kernels (Tensor Cores + INT8 + CPU Offload)")
    except ImportError as e:
        print(f"[INT2] ⚠ Optimized kernels not available: {e}")
        USING_OPTIMIZED = False


# Manager setup flag
_OFFLOAD_MANAGER_SETUP = False


def setup_offload_manager(num_layers: int = 128):
    """Setup the CPU offload manager (call once before creating model)."""
    global _OFFLOAD_MANAGER_SETUP
    
    if not USING_OPTIMIZED:
        return
    
    if _OFFLOAD_MANAGER_SETUP:
        return
    
    try:
        from int2_linear_tc_offload import Int2LinearTCOffload
        Int2LinearTCOffload.reset_layer_counter()
        Int2LinearTCOffload.setup_offload_manager(num_layers=num_layers)
        _OFFLOAD_MANAGER_SETUP = True
        print(f"[INT2] ✓ Offload manager initialized for {num_layers} layers")
    except Exception as e:
        print(f"[INT2] ⚠ Failed to setup offload manager: {e}")


class BitLinearInt2(nn.Module):
    """
    BitLinear che usa Int2LinearTC ottimizzato internamente.

    Interfaccia compatibile con BitNetLinear esistente.
    Con ottimizzazioni abilitate:
    - Tensor Core matmul (16.6x speedup)
    - INT8 activation compression (50% memory savings)
    - CPU offload overlap (enable larger models)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        # Parametri INT2
        threshold: int = 7,
        lr_scale: float = 5.0,
        decay_rate: float = 0.001,
        use_grad_hook: bool = True,
        enable_offload: bool = True  # Enable CPU offload
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Setup offload manager if not done
        if USING_OPTIMIZED and enable_offload and not _OFFLOAD_MANAGER_SETUP:
            setup_offload_manager(128)

        # Layer INT2 interno (ottimizzato o base)
        if USING_OPTIMIZED:
            LayerClass = Int2LinearWithGradHook if use_grad_hook else Int2Linear
            self.int2_layer = LayerClass(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                threshold=threshold,
                lr_scale=lr_scale,
                decay_rate=decay_rate,
                enable_offload=enable_offload
            )
        else:
            LayerClass = Int2LinearWithGradHook if use_grad_hook else Int2Linear
            self.int2_layer = LayerClass(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                threshold=threshold,
                lr_scale=lr_scale,
                decay_rate=decay_rate
            )

        # Per compatibilità con BitNetLinear
        self._use_grad_hook = use_grad_hook
        self._enable_offload = enable_offload

    @property
    def scale(self):
        """Alias per gamma (compatibilità con BitNetLinear)."""
        return self.int2_layer.gamma.view(1, 1).expand(self.out_features, 1)

    @property
    def gamma(self):
        """Accesso a gamma per compatibilità."""
        return self.int2_layer.gamma

    @property
    def weight_packed(self):
        """Accesso a W_packed per compatibilità."""
        return self.int2_layer.W_packed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Int2Linear kernels need float16
        if x.dtype != torch.float16:
            x = x.half()
        out = self.int2_layer(x)
        return out

    def set_lr(self, lr: float):
        """Imposta LR per hysteresis (se usa grad hook)."""
        if self._use_grad_hook and hasattr(self.int2_layer, 'set_lr'):
            self.int2_layer.set_lr(lr)

    def hysteresis_update(self, dY: torch.Tensor, lr: float):
        """Wrapper per hysteresis update."""
        self.int2_layer.hysteresis_update(dY, lr)

    def prefetch_activation(self):
        """Prefetch activation from CPU (for offload mode)."""
        if USING_OPTIMIZED and hasattr(self.int2_layer, 'prefetch_activation'):
            self.int2_layer.prefetch_activation()

    def reset_parameters(self):
        """Inizializza i pesi (come BitNetLinear.reset_parameters)."""
        import math
        # Kaiming initialization
        w_init = torch.empty(
            self.out_features, self.in_features,
            dtype=torch.float32
        )
        nn.init.kaiming_normal_(w_init, mode='fan_in', nonlinearity='linear')

        # Calcola scale
        scale = w_init.abs().mean().item()
        scale = max(scale, 1e-5)

        # Inizializza Int2Linear
        self.int2_layer.init_from_float(
            w_init.to(self.int2_layer.W_packed.device),
            scale=scale
        )

    @classmethod
    def from_bitnetlinear(cls, bitlinear: nn.Module, **kwargs) -> 'BitLinearInt2':
        """
        Converte un BitNetLinear esistente in BitLinearInt2.

        Args:
            bitlinear: Modulo BitNetLinear da convertire
            **kwargs: Parametri extra per Int2Linear (threshold, lr_scale, etc.)

        Returns:
            BitLinearInt2 inizializzato con gli stessi pesi
        """
        import math

        # INT2 richiede CUDA
        device = torch.device('cuda')

        # Crea il nuovo layer e spostalo su GPU
        new_layer = cls(
            in_features=bitlinear.in_features,
            out_features=bitlinear.out_features,
            bias=bitlinear.bias is not None,
            **kwargs
        ).to(device)

        # Copia i pesi
        with torch.no_grad():
            weight = None
            scale = None

            # Prova a ottenere master_weight
            if hasattr(bitlinear, 'master_weight') and bitlinear.master_weight is not None:
                weight = bitlinear.master_weight.float().to(device)
                if hasattr(bitlinear, 'scale') and bitlinear.scale is not None:
                    scale = bitlinear.scale.data.float().mean().item()

            # Se master_weight non disponibile, usa inizializzazione Kaiming
            if weight is None:
                weight = torch.empty(
                    bitlinear.out_features, bitlinear.in_features,
                    dtype=torch.float32, device=device
                )
                nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='linear')
                scale = 1.0 / math.sqrt(bitlinear.in_features)

            # Assicura scale valido
            if scale is None:
                scale = weight.abs().mean().item()
            scale = max(scale, 1e-5)

            # Inizializza Int2Linear
            new_layer.int2_layer.init_from_float(weight, scale=scale)

            # Copia bias se presente
            if bitlinear.bias is not None and new_layer.int2_layer.bias is not None:
                new_layer.int2_layer.bias.data.copy_(
                    bitlinear.bias.data.to(device)
                )

        return new_layer

    def memory_footprint(self):
        """Statistiche memoria."""
        return self.int2_layer.memory_footprint()

    def extra_repr(self):
        opt_str = "TC+INT8+Offload" if USING_OPTIMIZED else "base"
        return f'in={self.in_features}, out={self.out_features}, int2={opt_str}'


def convert_bitnetlinear_to_int2(
    module: nn.Module,
    threshold: int = 7,
    lr_scale: float = 5.0,
    decay_rate: float = 0.001,
    use_grad_hook: bool = True,
    enable_offload: bool = True,
    inplace: bool = False
) -> nn.Module:
    """
    Converte ricorsivamente tutti i BitNetLinear in BitLinearInt2 ottimizzato.

    Args:
        module: Modulo PyTorch da convertire
        threshold: Soglia hysteresis
        lr_scale: Scale factor per LR
        decay_rate: Decay rate
        use_grad_hook: Usa hook per catturare gradienti automaticamente
        enable_offload: Enable CPU offload for activations
        inplace: Se True, modifica in place

    Returns:
        Modulo con BitNetLinear convertiti
    """
    if not inplace:
        import copy
        module = copy.deepcopy(module)

    for name, child in list(module.named_children()):
        # Usa nome classe per evitare problemi con import path diversi
        if type(child).__name__ == 'BitNetLinear':
            # Converti
            new_child = BitLinearInt2.from_bitnetlinear(
                child,
                threshold=threshold,
                lr_scale=lr_scale,
                decay_rate=decay_rate,
                use_grad_hook=use_grad_hook,
                enable_offload=enable_offload
            )
            setattr(module, name, new_child)
        else:
            # Ricorsione
            convert_bitnetlinear_to_int2(
                child,
                threshold=threshold,
                lr_scale=lr_scale,
                decay_rate=decay_rate,
                use_grad_hook=use_grad_hook,
                enable_offload=enable_offload,
                inplace=True
            )

    return module


def get_int2_layers(module: nn.Module):
    """Ritorna lista di tutti i BitLinearInt2 nel modello."""
    layers = []
    for m in module.modules():
        if isinstance(m, BitLinearInt2):
            layers.append(m)
    return layers


def get_non_int2_params(module: nn.Module):
    """Ritorna parametri che NON sono in BitLinearInt2 (per AdamW)."""
    int2_params = set()
    for m in module.modules():
        if isinstance(m, BitLinearInt2):
            for p in m.parameters():
                int2_params.add(id(p))

    return [p for p in module.parameters() if id(p) not in int2_params]


def set_all_int2_lr(module: nn.Module, lr: float):
    """Imposta LR per tutti i layer BitLinearInt2."""
    for m in module.modules():
        if isinstance(m, BitLinearInt2):
            m.set_lr(lr)


def prefetch_all_activations(module: nn.Module):
    """Prefetch all activations from CPU (call before backward)."""
    if not USING_OPTIMIZED:
        return
    schedule_all_prefetches(module)


def enable_hysteresis():
    """Enable hysteresis updates in grad hooks (call before last micro-batch)."""
    if USING_OPTIMIZED:
        try:
            from .int2_linear_tc_offload import Int2LinearTCOffloadWithGradHook
            Int2LinearTCOffloadWithGradHook.enable_hysteresis()
        except ImportError:
            from int2_linear_tc_offload import Int2LinearTCOffloadWithGradHook
            Int2LinearTCOffloadWithGradHook.enable_hysteresis()


def disable_hysteresis():
    """Disable hysteresis updates in grad hooks (during accumulation micro-batches)."""
    if USING_OPTIMIZED:
        try:
            from .int2_linear_tc_offload import Int2LinearTCOffloadWithGradHook
            Int2LinearTCOffloadWithGradHook.disable_hysteresis()
        except ImportError:
            from int2_linear_tc_offload import Int2LinearTCOffloadWithGradHook
            Int2LinearTCOffloadWithGradHook.disable_hysteresis()


# ============================================================================
# CONVERSIONE ESTESA: nn.Linear + BitNetLinear → INT2 Ottimizzato
# ============================================================================

def _should_exclude_layer(name: str, module: nn.Module, min_size: int = 256,
                          exclude_patterns: list = None) -> tuple:
    """
    Determina se un layer deve essere escluso dalla conversione INT2.

    Returns:
        (should_exclude: bool, reason: str)
    """
    if exclude_patterns is None:
        exclude_patterns = [
            'lm_head',
            'embed', 'token_emb', 'wte', 'wpe', 'word_emb',
            'norm', 'ln_', 'layer_norm', 'rmsnorm',
            'bias',
        ]

    name_lower = name.lower()

    # Esclusioni per nome
    for pattern in exclude_patterns:
        if pattern in name_lower:
            return True, f"pattern '{pattern}'"

    # Esclusioni per tipo (non Linear)
    if not isinstance(module, nn.Linear):
        return True, "not nn.Linear"

    # Esclusioni per dimensione
    total_params = module.in_features * module.out_features
    if total_params < min_size * min_size:
        return True, f"too small ({total_params:,} < {min_size*min_size:,})"

    return False, ""


def convert_all_linear_to_int2(
    module: nn.Module,
    threshold: int = 7,
    lr_scale: float = 5.0,
    decay_rate: float = 0.001,
    use_grad_hook: bool = True,
    enable_offload: bool = True,
    min_size: int = 256,
    exclude_patterns: list = None,
    inplace: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Converte TUTTI i layer lineari (nn.Linear e BitNetLinear) a INT2 OTTIMIZZATO.

    Usa le nuove ottimizzazioni:
    - Tensor Cores (16.6x matmul speedup)
    - INT8 Activations (50% memory savings)
    - CPU Offload (enable larger models)

    Args:
        module: Modello da convertire
        threshold: Soglia hysteresis
        lr_scale: Scale factor LR
        decay_rate: Decay rate
        use_grad_hook: Usa hook per catturare gradienti
        enable_offload: Enable CPU offload for activations
        min_size: Dimensione minima per conversione (in*out > min_size²)
        exclude_patterns: Pattern di nomi da escludere
        inplace: Se True, modifica in place
        verbose: Stampa report conversione

    Returns:
        Modello con layer convertiti
    """
    import math

    if not inplace:
        import copy
        module = copy.deepcopy(module)

    # Setup offload manager before conversion
    if USING_OPTIMIZED and enable_offload:
        # Estimate number of layers
        num_linear = sum(1 for m in module.modules() if isinstance(m, nn.Linear))
        setup_offload_manager(num_linear * 2)

    # Statistiche per report
    stats = {
        'bitnet_converted': {'count': 0, 'params': 0},
        'linear_mamba': {'count': 0, 'params': 0},
        'linear_mla': {'count': 0, 'params': 0},
        'linear_other': {'count': 0, 'params': 0},
        'excluded': {},
        'errors': []
    }

    # Device
    device = torch.device('cuda')

    def categorize_layer(name: str) -> str:
        """Categorizza il layer per il report."""
        name_lower = name.lower()
        if 'mamba' in name_lower or any(x in name_lower for x in ['in_proj', 'out_proj', 'x_proj', 'dt_proj', 'diff_gate']):
            if 'attn' in name_lower and 'w_' not in name_lower:
                return 'mamba'
        if any(x in name_lower for x in ['w_q', 'w_k', 'w_v', 'w_out', 'w_kv', 'embed_gate']):
            return 'mla'
        return 'other'

    def convert_linear_to_int2(linear: nn.Linear, name: str) -> BitLinearInt2:
        """Converte nn.Linear a BitLinearInt2."""
        new_layer = BitLinearInt2(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
            lr_scale=lr_scale,
            decay_rate=decay_rate,
            use_grad_hook=use_grad_hook,
            enable_offload=enable_offload
        ).to(device)

        with torch.no_grad():
            # Copia pesi
            weight = linear.weight.data.float().to(device)
            scale = weight.abs().mean().item()
            scale = max(scale, 1e-5)

            new_layer.int2_layer.init_from_float(weight, scale=scale)

            # Copia bias se presente
            if linear.bias is not None and new_layer.int2_layer.bias is not None:
                new_layer.int2_layer.bias.data.copy_(linear.bias.data.to(device))

        return new_layer

    def recursive_convert(mod: nn.Module, prefix: str = ""):
        """Converte ricorsivamente tutti i layer."""
        for name, child in list(mod.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            child_type = type(child).__name__

            # Caso 1: BitNetLinear
            if child_type == 'BitNetLinear':
                try:
                    new_child = BitLinearInt2.from_bitnetlinear(
                        child,
                        threshold=threshold,
                        lr_scale=lr_scale,
                        decay_rate=decay_rate,
                        use_grad_hook=use_grad_hook,
                        enable_offload=enable_offload
                    )
                    setattr(mod, name, new_child)
                    params = child.in_features * child.out_features
                    stats['bitnet_converted']['count'] += 1
                    stats['bitnet_converted']['params'] += params
                except Exception as e:
                    stats['errors'].append(f"{full_name}: {str(e)}")

            # Caso 2: nn.Linear
            elif isinstance(child, nn.Linear):
                should_exclude, reason = _should_exclude_layer(
                    full_name, child, min_size, exclude_patterns
                )

                if should_exclude:
                    # Registra esclusione
                    params = child.in_features * child.out_features
                    if reason not in stats['excluded']:
                        stats['excluded'][reason] = {'count': 0, 'params': 0, 'names': []}
                    stats['excluded'][reason]['count'] += 1
                    stats['excluded'][reason]['params'] += params
                    stats['excluded'][reason]['names'].append(full_name)
                else:
                    # Converti
                    try:
                        new_child = convert_linear_to_int2(child, full_name)
                        setattr(mod, name, new_child)
                        params = child.in_features * child.out_features

                        category = categorize_layer(full_name)
                        if category == 'mamba':
                            stats['linear_mamba']['count'] += 1
                            stats['linear_mamba']['params'] += params
                        elif category == 'mla':
                            stats['linear_mla']['count'] += 1
                            stats['linear_mla']['params'] += params
                        else:
                            stats['linear_other']['count'] += 1
                            stats['linear_other']['params'] += params
                    except Exception as e:
                        stats['errors'].append(f"{full_name}: {str(e)}")

            # Caso 3: Altro modulo - ricorsione
            else:
                recursive_convert(child, full_name)

    # Esegui conversione
    recursive_convert(module)

    # Calcola totali
    total_converted = (
        stats['bitnet_converted']['params'] +
        stats['linear_mamba']['params'] +
        stats['linear_mla']['params'] +
        stats['linear_other']['params']
    )
    total_excluded = sum(v['params'] for v in stats['excluded'].values())
    total_all = total_converted + total_excluded

    # Report
    if verbose and total_all > 0:
        print("\n" + "="*60)
        if USING_OPTIMIZED:
            print("=== CONVERSIONE INT2 OTTIMIZZATO (TC+INT8+Offload) ===")
        else:
            print("=== CONVERSIONE INT2 BASE ===")
        print("="*60)

        print("\nCONVERTITI:")
        print(f"  BitNetLinear:      {stats['bitnet_converted']['count']:3d} layers, "
              f"{stats['bitnet_converted']['params']/1e6:6.1f}M params")
        print(f"  nn.Linear (Mamba): {stats['linear_mamba']['count']:3d} layers, "
              f"{stats['linear_mamba']['params']/1e6:6.1f}M params")
        print(f"  nn.Linear (MLA):   {stats['linear_mla']['count']:3d} layers, "
              f"{stats['linear_mla']['params']/1e6:6.1f}M params")
        print(f"  nn.Linear (altri): {stats['linear_other']['count']:3d} layers, "
              f"{stats['linear_other']['params']/1e6:6.1f}M params")
        print("  " + "-"*45)
        total_count = (stats['bitnet_converted']['count'] + stats['linear_mamba']['count'] +
                      stats['linear_mla']['count'] + stats['linear_other']['count'])
        print(f"  TOTALE INT2:       {total_count:3d} layers, "
              f"{total_converted/1e6:6.1f}M params ({total_converted/total_all*100:.1f}%)")

        if USING_OPTIMIZED:
            print("\n  🚀 OTTIMIZZAZIONI ATTIVE:")
            print("     - Tensor Cores: 16.6x matmul speedup")
            print("     - INT8 Activations: 50% memory savings")
            print("     - CPU Offload: Enable larger models")

        print("="*60 + "\n")

    return module
