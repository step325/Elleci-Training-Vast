"""
Overlap Transfer Manager for CPU Offloading with Compute Overlap.

This module provides asynchronous GPU↔CPU memory transfer capabilities
that overlap with GPU compute operations to hide transfer latency.

Phase 4 of INT2 Optimizations.
"""

import torch
import torch.cuda
from typing import Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
import threading


@dataclass
class OffloadState:
    """State for a single offloaded tensor."""
    cpu_data: torch.Tensor  # Pinned CPU tensor
    cpu_scale: torch.Tensor  # Scale value (for INT8 quantized data)
    original_shape: tuple
    dtype: torch.dtype
    offload_event: Optional[torch.cuda.Event] = None
    prefetch_event: Optional[torch.cuda.Event] = None
    gpu_staging: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


class OverlapTransferManager:
    """
    Manages asynchronous CPU offloading with compute overlap.

    This manager enables training larger models by offloading saved activations
    to CPU memory while hiding transfer latency through overlap with GPU compute.

    Key Features:
    - Separate CUDA stream for transfers (doesn't block compute)
    - Pinned memory for fast GPU↔CPU transfers
    - Async prefetch to hide CPU→GPU latency during backward
    - Buffer reuse to minimize allocations

    Usage:
        # Setup
        manager = OverlapTransferManager(num_layers=32, device='cuda')

        # Forward pass - offload activations
        for layer_id, activation in enumerate(activations):
            manager.offload_async(layer_id, activation)

        # Before backward - schedule prefetches
        for layer_id in reversed(range(num_layers)):
            manager.prefetch_async(layer_id)

        # Backward pass - retrieve activations
        for layer_id in reversed(range(num_layers)):
            activation = manager.retrieve(layer_id)
            # Use activation for gradient computation

    Thread Safety:
        This class is NOT thread-safe. Use one manager per training process.
    """

    def __init__(
        self,
        num_layers: int = 128,
        device: torch.device = None,
        preallocate: bool = False,
        max_buffer_bytes: int = 0
    ):
        """
        Initialize the overlap transfer manager.

        Args:
            num_layers: Maximum number of layers to track
            device: CUDA device for GPU operations
            preallocate: If True, pre-allocate CPU buffers at init
            max_buffer_bytes: Max bytes per layer (for preallocation)
        """
        self.device = device or torch.device('cuda')
        self.num_layers = num_layers

        # Create dedicated transfer stream
        self._transfer_stream = torch.cuda.Stream(device=self.device)

        # Layer state storage
        self._states: Dict[int, OffloadState] = {}

        # Tracking sets
        self._offloaded: Set[int] = set()
        self._prefetching: Set[int] = set()

        # Statistics
        self._stats = {
            'offload_count': 0,
            'prefetch_count': 0,
            'sync_fallback_count': 0,  # Times we had to do sync copy
            'total_bytes_transferred': 0,
        }

        # Pre-allocation if requested
        self._preallocated = preallocate
        if preallocate and max_buffer_bytes > 0:
            self._preallocate_buffers(num_layers, max_buffer_bytes)

    def _preallocate_buffers(self, num_layers: int, max_bytes: int):
        """Pre-allocate pinned CPU buffers."""
        for layer_id in range(num_layers):
            # Allocate as raw bytes, will be reshaped later
            self._states[layer_id] = OffloadState(
                cpu_data=torch.empty(max_bytes, dtype=torch.int8, pin_memory=True),
                cpu_scale=torch.empty(1, dtype=torch.float32, pin_memory=True),
                original_shape=None,
                dtype=torch.int8,
            )

    def offload_async(
        self,
        layer_id: int,
        tensor: Tuple[torch.Tensor, torch.Tensor],
        sync_point: bool = False
    ) -> None:
        """
        Asynchronously offload INT8-compressed activation to CPU.

        This operation runs on the transfer stream and doesn't block compute.
        The GPU memory can be freed immediately after this call.

        IMPORTANT: We must synchronize with the compute stream before copying,
        because int8_data was produced on the compute stream. Without this,
        the transfer stream may read incomplete/stale data (race condition).

        Args:
            layer_id: Unique identifier for this layer
            tensor: Tuple of (int8_data, scale) from quantize_activation
            sync_point: If True, record event for later synchronization
        """
        int8_data, scale = tensor

        # Ensure we have a state entry
        if layer_id not in self._states:
            # Allocate pinned CPU buffers
            cpu_data = torch.empty(
                int8_data.shape,
                dtype=torch.int8,
                pin_memory=True
            )
            cpu_scale = torch.empty(1, dtype=torch.float32, pin_memory=True)

            self._states[layer_id] = OffloadState(
                cpu_data=cpu_data,
                cpu_scale=cpu_scale,
                original_shape=int8_data.shape,
                dtype=int8_data.dtype,
            )
        else:
            state = self._states[layer_id]
            # Reallocate if shape changed
            if state.original_shape != int8_data.shape:
                state.cpu_data = torch.empty(
                    int8_data.shape,
                    dtype=torch.int8,
                    pin_memory=True
                )
                state.original_shape = int8_data.shape

        state = self._states[layer_id]

        # Record event on the compute stream so the transfer stream can
        # wait for the data to be fully produced before copying it.
        compute_event = torch.cuda.current_stream().record_event()

        # Schedule async copy on transfer stream
        with torch.cuda.stream(self._transfer_stream):
            # Wait for compute stream to finish producing int8_data
            self._transfer_stream.wait_event(compute_event)

            state.cpu_data.copy_(int8_data, non_blocking=True)
            # Scale: copy to CPU (synchronous on transfer stream, but it's 4 bytes)
            if scale.is_cuda:
                state.cpu_scale.copy_(scale.cpu(), non_blocking=True)
            else:
                state.cpu_scale.copy_(scale, non_blocking=True)

        # Record completion event
        state.offload_event = self._transfer_stream.record_event()

        # Track state
        self._offloaded.add(layer_id)
        self._stats['offload_count'] += 1
        self._stats['total_bytes_transferred'] += int8_data.numel()

    def prefetch_async(
        self,
        layer_id: int,
        target_device: torch.device = None
    ) -> None:
        """
        Asynchronously prefetch activation back to GPU.

        Call this BEFORE you need the data, ideally during compute of
        another layer to overlap transfer with compute.

        Args:
            layer_id: Layer identifier to prefetch
            target_device: GPU device to prefetch to
        """
        if layer_id not in self._offloaded:
            return  # Nothing to prefetch

        if layer_id in self._prefetching:
            return  # Already prefetching

        target_device = target_device or self.device
        state = self._states[layer_id]

        # Wait for offload to complete before reading CPU buffer
        if state.offload_event is not None:
            self._transfer_stream.wait_event(state.offload_event)

        # Allocate GPU staging buffers
        with torch.cuda.stream(self._transfer_stream):
            gpu_data = torch.empty(
                state.cpu_data.shape,
                dtype=torch.int8,
                device=target_device
            )
            gpu_scale = torch.empty(1, dtype=torch.float32, device=target_device)

            # Async copy CPU→GPU
            gpu_data.copy_(state.cpu_data, non_blocking=True)
            gpu_scale.copy_(state.cpu_scale, non_blocking=True)

            state.gpu_staging = (gpu_data, gpu_scale)

        # Record completion event
        state.prefetch_event = self._transfer_stream.record_event()

        self._prefetching.add(layer_id)
        self._stats['prefetch_count'] += 1

    def retrieve(
        self,
        layer_id: int,
        wait: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the activation back from CPU.

        If prefetch_async was called earlier, this uses the prefetched data.
        Otherwise, performs a synchronous transfer (slower).

        Args:
            layer_id: Layer identifier
            wait: If True, wait for any in-flight transfer to complete

        Returns:
            Tuple of (int8_data, scale) on GPU
        """
        if layer_id not in self._offloaded:
            raise RuntimeError(f"Layer {layer_id} not offloaded")

        state = self._states[layer_id]

        if layer_id in self._prefetching and state.gpu_staging is not None:
            # Use prefetched data
            if wait and state.prefetch_event is not None:
                # Wait on current compute stream for prefetch to complete
                torch.cuda.current_stream().wait_event(state.prefetch_event)

            result = state.gpu_staging
            state.gpu_staging = None
            state.prefetch_event = None
            self._prefetching.discard(layer_id)

        else:
            # Synchronous fallback - prefetch wasn't called
            self._stats['sync_fallback_count'] += 1

            # Wait for offload to complete
            if state.offload_event is not None:
                state.offload_event.synchronize()

            # Sync copy
            gpu_data = state.cpu_data.to(self.device, non_blocking=False)
            gpu_scale = state.cpu_scale.to(self.device, non_blocking=False)
            result = (gpu_data, gpu_scale)

        # Clear offload tracking
        self._offloaded.discard(layer_id)

        return result

    def clear(self, layer_id: int) -> None:
        """
        Clear all state for a layer.

        Call after hysteresis update to free any remaining resources.
        """
        if layer_id in self._states:
            state = self._states[layer_id]
            state.gpu_staging = None
            state.offload_event = None
            state.prefetch_event = None

        self._offloaded.discard(layer_id)
        self._prefetching.discard(layer_id)

    def clear_all(self) -> None:
        """Clear all layer states."""
        for layer_id in list(self._states.keys()):
            self.clear(layer_id)

    def synchronize(self) -> None:
        """Wait for all pending transfers to complete."""
        self._transfer_stream.synchronize()

    def is_offloaded(self, layer_id: int) -> bool:
        """Check if a layer's activation is currently offloaded."""
        return layer_id in self._offloaded

    @property
    def stats(self) -> Dict[str, Any]:
        """Get transfer statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            'offload_count': 0,
            'prefetch_count': 0,
            'sync_fallback_count': 0,
            'total_bytes_transferred': 0,
        }

    def memory_usage(self) -> Dict[str, int]:
        """Get current memory usage."""
        cpu_bytes = 0
        gpu_staging_bytes = 0

        for state in self._states.values():
            if state.cpu_data is not None:
                cpu_bytes += state.cpu_data.numel() * state.cpu_data.element_size()
            if state.gpu_staging is not None:
                gpu_staging_bytes += state.gpu_staging[0].numel()

        return {
            'cpu_pinned_bytes': cpu_bytes,
            'gpu_staging_bytes': gpu_staging_bytes,
            'num_offloaded': len(self._offloaded),
            'num_prefetching': len(self._prefetching),
        }


class LayerOffloadContext:
    """
    Context manager for automatic offload/prefetch of a group of layers.

    Usage:
        with LayerOffloadContext(manager, layer_ids) as ctx:
            # Forward pass
            for i, layer in enumerate(layers):
                output = layer(input)
                ctx.offload_forward(i, layer._saved_input)
                input = output

            # Start prefetch before backward
            ctx.start_prefetch()

            # Backward pass
            loss.backward()
            for i in reversed(range(len(layers))):
                activation = ctx.get_for_backward(i)
                # ... hysteresis update
    """

    def __init__(
        self,
        manager: OverlapTransferManager,
        layer_ids: list
    ):
        self.manager = manager
        self.layer_ids = layer_ids
        self._forward_order = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Clear all on exit
        for layer_id in self.layer_ids:
            self.manager.clear(layer_id)

    def offload_forward(self, idx: int, tensor: Tuple[torch.Tensor, torch.Tensor]):
        """Offload activation during forward pass."""
        layer_id = self.layer_ids[idx]
        self.manager.offload_async(layer_id, tensor)
        self._forward_order.append(layer_id)

    def start_prefetch(self):
        """Start prefetching all layers in reverse order."""
        for layer_id in reversed(self._forward_order):
            self.manager.prefetch_async(layer_id)

    def get_for_backward(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get activation for backward pass."""
        layer_id = self.layer_ids[idx]
        return self.manager.retrieve(layer_id)


# Global manager instance (optional, for convenience)
_global_manager: Optional[OverlapTransferManager] = None


def get_global_manager() -> Optional[OverlapTransferManager]:
    """Get the global transfer manager."""
    return _global_manager


def setup_global_manager(**kwargs) -> OverlapTransferManager:
    """Setup and return the global transfer manager."""
    global _global_manager
    _global_manager = OverlapTransferManager(**kwargs)
    return _global_manager


def cleanup_global_manager():
    """Cleanup the global manager."""
    global _global_manager
    if _global_manager is not None:
        _global_manager.synchronize()
        _global_manager.clear_all()
        _global_manager = None
