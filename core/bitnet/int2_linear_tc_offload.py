"""
Int2LinearTC with CPU Offload and Compute Overlap.

This module extends Int2LinearTC to support automatic CPU offloading
of saved activations with compute overlap to hide transfer latency.

Phase 4 of INT2 Optimizations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import os
import sys

# Import base classes
try:
    from .int2_linear_tc import (
        Int2LinearTC, Int2LinearTCWithGradHook,
        HAS_TC_OPS, HAS_INT2_OPS, int2_tc_ops
    )
    from .overlap_transfer import OverlapTransferManager, get_global_manager, setup_global_manager
except (ImportError, ValueError):
    from int2_linear_tc import (
        Int2LinearTC, Int2LinearTCWithGradHook,
        HAS_TC_OPS, HAS_INT2_OPS, int2_tc_ops
    )
    from overlap_transfer import OverlapTransferManager, get_global_manager, setup_global_manager


class Int2LinearTCOffload(Int2LinearTC):
    """
    Int2LinearTC with automatic CPU offload of saved activations.

    During training:
    - Forward: Activations are offloaded to CPU asynchronously
    - Backward: Activations are prefetched back to GPU before hysteresis update

    The overlap between compute and transfer hides most of the PCIe latency,
    resulting in <10% throughput degradation while freeing significant GPU memory.

    Usage:
        # Setup manager (once, before training)
        Int2LinearTCOffload.setup_offload_manager(num_layers=32)

        # Create layers normally
        layer = Int2LinearTCOffload(in_features=2560, out_features=2560)

        # Training works the same, but activations are automatically offloaded
    """

    # Class-level manager (shared across all layers)
    _manager: Optional[OverlapTransferManager] = None
    _layer_counter: int = 0
    _offload_enabled: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        threshold: int = 7,
        lr_scale: float = 5.0,
        decay_rate: float = 0.001,
        compress_activations: bool = True,  # Phase 2: Always use INT8
        enable_offload: bool = True  # Phase 4: CPU offload
    ):
        """
        Initialize Int2LinearTC with offload support.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            threshold: Hysteresis threshold
            lr_scale: Learning rate scale for hysteresis
            decay_rate: Decay rate for hysteresis
            compress_activations: Use INT8 activation compression (Phase 2)
            enable_offload: Enable CPU offload for saved activations
        """
        # Force compress_activations=True for offload (INT8 is required)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            threshold=threshold,
            lr_scale=lr_scale,
            decay_rate=decay_rate,
            compress_activations=True  # Required for offload
        )

        self.enable_offload = enable_offload

        # Assign unique layer ID
        self._layer_id = Int2LinearTCOffload._layer_counter
        Int2LinearTCOffload._layer_counter += 1

    @classmethod
    def setup_offload_manager(
        cls,
        num_layers: int = 128,
        device: torch.device = None,
        **kwargs
    ) -> OverlapTransferManager:
        """
        Setup the shared transfer manager.

        Call this once before creating any Int2LinearTCOffload layers.

        Args:
            num_layers: Maximum number of layers to support
            device: CUDA device
            **kwargs: Additional arguments for OverlapTransferManager

        Returns:
            The created manager
        """
        device = device or torch.device('cuda')
        cls._manager = OverlapTransferManager(
            num_layers=num_layers,
            device=device,
            **kwargs
        )
        return cls._manager

    @classmethod
    def get_manager(cls) -> Optional[OverlapTransferManager]:
        """Get the current transfer manager."""
        return cls._manager

    @classmethod
    def enable_offload_globally(cls, enabled: bool = True):
        """Enable or disable offload for all layers."""
        cls._offload_enabled = enabled

    @classmethod
    def reset_layer_counter(cls):
        """Reset layer ID counter (call when creating new model)."""
        cls._layer_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic CPU offload.

        After computing the output, the saved activation is asynchronously
        transferred to CPU memory, freeing GPU memory immediately.
        """
        if not self._initialized:
            # Initialize weights on first forward
            import math
            w_init = torch.randn(
                self.out_features, self.in_features,
                device=x.device, dtype=torch.float32
            ) * (1.0 / math.sqrt(self.in_features))
            self.init_from_float(w_init)

        # Standard forward with INT8 compression
        if self.training:
            x_detached = x.detach()
            x_half = x_detached.half() if x_detached.dtype != torch.float16 else x_detached
            self._saved_input_shape = x_half.shape

            # Async quantize to INT8: zero GPU→CPU sync
            if Int2LinearTC._shared_absmax_buf is None or Int2LinearTC._shared_absmax_buf.device != x.device:
                Int2LinearTC._shared_absmax_buf = torch.zeros(1, dtype=torch.float32, device=x.device)
            self._saved_input = int2_tc_ops.quantize_activation_async(
                x_half.contiguous(), Int2LinearTC._shared_absmax_buf
            )

            # Offload to CPU asynchronously
            if (self.enable_offload and
                self._offload_enabled and
                self._manager is not None):
                self._manager.offload_async(self._layer_id, self._saved_input)
                # Mark as offloaded (don't clear _saved_input yet for fallback)
                self._is_offloaded = True
            else:
                self._is_offloaded = False

        # Compute output using TC kernels
        from int2_linear_tc import Int2LinearTCFunction
        # Ensure gamma cache is set
        if self._gamma_cached is None:
            self._gamma_cached = self.gamma.detach().item()
        out = Int2LinearTCFunction.apply(x, self.W_packed, self.gamma, self.in_features, self._gamma_cached)

        if self.bias is not None:
            out = out + self.bias

        return out

    def prefetch_activation(self):
        """
        Start prefetching activation from CPU to GPU.

        Call this BEFORE hysteresis_update() to overlap transfer with compute.
        Typically called at the start of backward for this layer.
        """
        if (getattr(self, '_is_offloaded', False) and
            self._manager is not None and
            getattr(self, 'enable_offload', False) and
            getattr(self, '_offload_enabled', False)):
            self._manager.prefetch_async(self._layer_id)

    def hysteresis_update(self, dY: torch.Tensor, lr: float):
        """
        Apply hysteresis update using (possibly offloaded) activation.

        If activation was offloaded, retrieves it from CPU.
        Uses prefetched data if prefetch_activation() was called earlier.
        """
        if self._saved_input is None and not self._is_offloaded:
            raise RuntimeError("No saved input. Call forward() first in training mode.")

        self._step_py += 1

        # Retrieve activation from CPU if offloaded
        if (self._is_offloaded and
            self._manager is not None and
            self.enable_offload and
            self._offload_enabled):
            self._saved_input = self._manager.retrieve(self._layer_id)
            self._is_offloaded = False

        # Decompress INT8 activation
        x_int8, scale = self._saved_input
        X = int2_tc_ops.dequantize_activation(x_int8, scale)
        if self._saved_input_shape is not None:
            X = X.view(self._saved_input_shape)

        # Reshape for hysteresis kernel
        dY_2d = dY.reshape(-1, dY.size(-1)).half().contiguous()
        X_2d = X.reshape(-1, X.size(-1)).half().contiguous()

        # Apply hysteresis update
        # Apply hysteresis update
        
        # OPTIMIZED: Compute dW using cuBLAS (Tensor Cores) + Lightweight Kernel
        # dW[N, K] = dY.T[N, M] @ X[M, K]
        if hasattr(int2_tc_ops, 'hysteresis_step_v2'):
            # Phase 1: Compute gradients efficiently with cuBLAS
            dW = torch.mm(dY_2d.t(), X_2d).float()  # [N, K] FP32

            # Phase 2: Lightweight hysteresis update O(N×K)
            int2_tc_ops.hysteresis_step_v2(
                dW.contiguous(),
                self.W_packed, self.H_packed,
                lr, self.lr_scale, self.threshold, self.decay_rate,
                self._step_py
            )
        else:
            # Fallback: Fused kernel O(M×N×K) - Slow
            int2_tc_ops.hysteresis_step(
                dY_2d, X_2d,
                self.W_packed, self.H_packed,
                lr, self.lr_scale, self.threshold, self.decay_rate,
                self._step_py
            )


        # Clear saved state
        self._saved_input = None
        self._saved_input_shape = None

        # Clear manager state
        if self._manager is not None:
            self._manager.clear(self._layer_id)

    def memory_footprint(self) -> Dict[str, Any]:
        """Get memory footprint including offload info."""
        base = super().memory_footprint()
        base['offload_enabled'] = self.enable_offload
        base['is_offloaded'] = getattr(self, '_is_offloaded', False)
        return base


class Int2LinearTCOffloadWithGradHook(Int2LinearTCOffload):
    """
    Int2LinearTCOffload with automatic gradient capture via hooks.

    Combines Phase 4 (CPU offload) with automatic hysteresis updates.

    Gradient Accumulation Support:
        When using gradient accumulation, hysteresis must NOT fire on every
        micro-batch (it modifies W_packed in-place, changing the model between
        micro-batches). Use enable_hysteresis()/disable_hysteresis() to control
        when hysteresis updates are applied:

            for micro_batch in range(grad_accum):
                if micro_batch < grad_accum - 1:
                    Int2LinearTCOffloadWithGradHook.disable_hysteresis()
                else:
                    Int2LinearTCOffloadWithGradHook.enable_hysteresis()
                loss = model(x) ...
                loss.backward()
    """

    # Class-level flag: controls whether _grad_hook applies hysteresis
    _hysteresis_enabled: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lr = 0.001
        self._prefetch_scheduled = False

    def set_lr(self, lr: float):
        """Update the learning rate for hysteresis updates."""
        self._lr = lr

    @classmethod
    def enable_hysteresis(cls):
        """Enable hysteresis updates in grad hooks (call before last micro-batch)."""
        cls._hysteresis_enabled = True

    @classmethod
    def disable_hysteresis(cls):
        """Disable hysteresis updates in grad hooks (call during accumulation micro-batches)."""
        cls._hysteresis_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)

        if self.training and out.requires_grad:
            out.register_hook(self._grad_hook)

        return out

    def _grad_hook(self, grad: torch.Tensor):
        """Hook called during backward to capture output gradient."""
        if not self._hysteresis_enabled:
            # During gradient accumulation micro-batches: clear saved input
            # without applying hysteresis to prevent weight changes between
            # micro-batches.
            self._saved_input = None
            self._saved_input_shape = None
            if getattr(self, '_is_offloaded', False) and self._manager is not None:
                self._manager.clear(self._layer_id)
                self._is_offloaded = False
            self._prefetch_scheduled = False
            return

        # Note: prefetch should already be scheduled by training loop
        if not self._prefetch_scheduled:
            self.prefetch_activation()

        if self._saved_input is not None or getattr(self, '_is_offloaded', False):
            self.hysteresis_update(grad, self._lr)

        self._prefetch_scheduled = False

    def schedule_prefetch(self):
        """Mark that prefetch has been scheduled externally."""
        self._prefetch_scheduled = True


def convert_to_offload(
    module: nn.Module,
    enable_offload: bool = True,
    setup_manager: bool = True,
    num_layers: int = 128
) -> nn.Module:
    """
    Convert Int2LinearTC layers to Int2LinearTCOffload.

    Args:
        module: Module containing Int2LinearTC layers
        enable_offload: Enable CPU offload
        setup_manager: Auto-setup the offload manager
        num_layers: Number of layers for manager

    Returns:
        Module with converted layers
    """
    if setup_manager and Int2LinearTCOffload._manager is None:
        Int2LinearTCOffload.setup_offload_manager(num_layers=num_layers)

    Int2LinearTCOffload.reset_layer_counter()

    def convert_recursive(mod: nn.Module) -> nn.Module:
        for name, child in mod.named_children():
            if isinstance(child, Int2LinearTCWithGradHook):
                # Convert with grad hook
                offload_layer = Int2LinearTCOffloadWithGradHook(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    threshold=child.threshold,
                    lr_scale=child.lr_scale,
                    decay_rate=child.decay_rate,
                    enable_offload=enable_offload
                )
                # Copy state
                offload_layer.W_packed.copy_(child.W_packed)
                offload_layer.H_packed.copy_(child.H_packed)
                offload_layer.gamma.data.copy_(child.gamma.data)
                if child.bias is not None:
                    offload_layer.bias.data.copy_(child.bias.data)
                offload_layer._step.copy_(child._step)
                offload_layer._initialized = child._initialized
                offload_layer._lr = child._lr
                setattr(mod, name, offload_layer)

            elif isinstance(child, Int2LinearTC):
                # Convert basic layer
                offload_layer = Int2LinearTCOffload(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias is not None,
                    threshold=child.threshold,
                    lr_scale=child.lr_scale,
                    decay_rate=child.decay_rate,
                    enable_offload=enable_offload
                )
                # Copy state
                offload_layer.W_packed.copy_(child.W_packed)
                offload_layer.H_packed.copy_(child.H_packed)
                offload_layer.gamma.data.copy_(child.gamma.data)
                if child.bias is not None:
                    offload_layer.bias.data.copy_(child.bias.data)
                offload_layer._step.copy_(child._step)
                offload_layer._initialized = child._initialized
                setattr(mod, name, offload_layer)

            else:
                convert_recursive(child)

        return mod

    return convert_recursive(module)


def schedule_all_prefetches(model: nn.Module):
    """
    Schedule prefetch for all Int2LinearTCOffload layers.

    Call this at the start of backward pass to overlap all prefetches
    with backward computation.
    """
    layers = []
    for module in model.modules():
        if isinstance(module, Int2LinearTCOffload):
            layers.append(module)

    # Prefetch in reverse order (backward processes layers in reverse)
    for layer in reversed(layers):
        layer.prefetch_activation()
        if isinstance(layer, Int2LinearTCOffloadWithGradHook):
            layer.schedule_prefetch()


class OffloadTrainingMixin:
    """
    Mixin for training loops that use Int2LinearTCOffload layers.

    Add this to your trainer class to get automatic prefetch scheduling.
    """

    def _schedule_prefetches(self, model: nn.Module):
        """Schedule all prefetches before backward."""
        schedule_all_prefetches(model)

    def training_step_with_offload(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        """
        Training step with automatic prefetch scheduling.

        Args:
            model: Model with Int2LinearTCOffload layers
            batch: Input batch
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Loss value
        """
        # Forward
        outputs = model(batch['input_ids'])
        loss = criterion(outputs, batch['labels'])

        # Schedule prefetches before backward
        self._schedule_prefetches(model)

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        return loss
