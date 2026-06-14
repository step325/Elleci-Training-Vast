
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
    ],
    key=['n_packed_bytes'],
)
@triton.jit
def unpack_2bit_kernel(
    packed_ptr,
    output_ptr,
    n_elements,
    n_packed_bytes,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Unpack int8 (packed 2-bit) weights to bfloat16 {-1, 0, 1}.
    
    Strategy: 1 thread = 1 packed byte = 4 outputs.
    
    Decode mapping via (val & 1) - (val >> 1):
        00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0
    """
    pid = tl.program_id(axis=0)
    
    # Byte indices for this block
    byte_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_byte = byte_idx < n_packed_bytes
    
    # Load packed bytes (coalesced, 1 per thread)
    packed = tl.load(packed_ptr + byte_idx, mask=mask_byte, other=0)
    
    # Extract 4 x 2-bit values
    v0 = (packed >> 0) & 3
    v1 = (packed >> 2) & 3
    v2 = (packed >> 4) & 3
    v3 = (packed >> 6) & 3
    
    # Branchless decode: (val & 1) - (val >> 1)
    d0 = ((v0 & 1) - (v0 >> 1)).to(tl.bfloat16)
    d1 = ((v1 & 1) - (v1 >> 1)).to(tl.bfloat16)
    d2 = ((v2 & 1) - (v2 >> 1)).to(tl.bfloat16)
    d3 = ((v3 & 1) - (v3 >> 1)).to(tl.bfloat16)
    
    # Output indices
    out_base = byte_idx * 4
    
    # Separate masks for each output (handles n_elements % 4 != 0)
    mask_o0 = (out_base + 0) < n_elements
    mask_o1 = (out_base + 1) < n_elements
    mask_o2 = (out_base + 2) < n_elements
    mask_o3 = (out_base + 3) < n_elements
    
    # Store 4 outputs per thread
    tl.store(output_ptr + out_base + 0, d0, mask=mask_byte & mask_o0)
    tl.store(output_ptr + out_base + 1, d1, mask=mask_byte & mask_o1)
    tl.store(output_ptr + out_base + 2, d2, mask=mask_byte & mask_o2)
    tl.store(output_ptr + out_base + 3, d3, mask=mask_byte & mask_o3)


def unpack_2bit_triton(packed_weight: torch.Tensor, output_shape: tuple) -> torch.Tensor:
    """
    Python wrapper for unpack kernel.
    
    Args:
        packed_weight: Int8 tensor [N, K // 4]
        output_shape: (N, K)
        
    Returns:
        Unpacked BF16 tensor [N, K]
    """
    assert packed_weight.dtype == torch.int8
    
    n_packed_bytes = packed_weight.numel()
    n_elements = 1
    for d in output_shape:
        n_elements *= d
    
    output = torch.empty(output_shape, device=packed_weight.device, dtype=torch.bfloat16)
    
    # Lambda grid for autotune: BLOCK_SIZE is selected dynamically
    grid = lambda meta: (triton.cdiv(n_packed_bytes, meta['BLOCK_SIZE']),)
    
    unpack_2bit_kernel[grid](
        packed_weight,
        output,
        n_elements,
        n_packed_bytes,
    )
    
    return output

