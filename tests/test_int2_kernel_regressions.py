"""
Regression tests for INT2 CUDA kernel invariants.

These tests intentionally inspect the CUDA sources because the critical bugs
are representation-level mistakes that can be caught without a GPU runtime.
"""
import re
from pathlib import Path


REPO = Path(__file__).parent.parent
KERNELS = REPO / "kernels" / "int2"


def _source(name: str) -> str:
    return (KERNELS / name).read_text()


def test_int2_reserved_encoding_is_not_decoded_as_plus_two():
    """The reserved 2-bit code 11 must not be decoded by raw bits - 1."""
    kernel_files = [
        "int2_packed.cuh",
        "int2_matmul_tc.cu",
        "int2_matmul_int8_tc_v2.cu",
        "int2_backward_tc.cu",
        "int2_backward_int8_tc_v2.cu",
        "int2_unpack.cu",
    ]

    raw_decode = re.compile(r"&\s*0x0?3\s*\)+\s*-\s*1")
    offenders = [
        name
        for name in kernel_files
        if raw_decode.search(_source(name).replace("\n", " "))
    ]

    assert offenders == [], (
        "reserved INT2 code 11 would decode to +2 via raw bits - 1 in "
        + ", ".join(offenders)
    )


def test_hysteresis_kernels_use_atomic_packed_writes():
    """Each thread updates one weight, so packed byte writes must be atomic."""
    for name in ["int2_hysteresis_tc.cu", "int2_hysteresis_v2.cu"]:
        src = _source(name)
        assert "pack_int2_unsafe" not in src
        assert "pack_int4_unsafe" not in src
        assert "bitpack::pack_int2(" in src
        assert "bitpack::pack_int4(" in src


def test_int2_cuda_wrappers_check_runtime_errors():
    """Host wrappers should not ignore CUDA allocation, copy, sync, or device errors."""
    checked_files = [
        "int2_activation_quant.cu",
        "int2_matmul_int8_tc_v2.cu",
        "int2_backward_int8_tc_v2.cu",
    ]

    for name in checked_files:
        src = _source(name)
        assert "CUDA_CHECK" in src, f"{name} has unchecked CUDA API calls"


def test_device_property_cache_uses_current_cuda_device():
    """Kernel selection must query the active device, not always device 0."""
    for name in ["int2_matmul_int8_tc_v2.cu", "int2_backward_int8_tc_v2.cu"]:
        src = _source(name)
        assert "cudaGetDevice(&device)" in src
        assert "cudaGetDeviceProperties(&prop, device)" in src
