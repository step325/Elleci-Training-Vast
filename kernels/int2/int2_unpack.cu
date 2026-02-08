/**
 * INT2 Unpack Kernel: W_packed (uint8) → W_fp16 (half)
 *
 * Each byte contains 4 INT2 weights encoded as:
 *   00 = -1, 01 = 0, 10 = +1, 11 = reserved (treated as 0)
 *
 * This kernel is bandwidth-bound and very fast (~18μs for 4096×4096).
 * The unpacked FP16 tensor is then used with cuBLAS for matmul,
 * which is ~3x faster than our custom WMMA kernel.
 *
 * Target: NVIDIA Ada Lovelace (SM 8.9) - RTX 4070/4080/4090
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Each thread processes 4 bytes = 16 weights
#define BYTES_PER_THREAD 4
#define WEIGHTS_PER_BYTE 4
#define WEIGHTS_PER_THREAD (BYTES_PER_THREAD * WEIGHTS_PER_BYTE)  // 16

#define BLOCK_SIZE 256

// Lookup table in constant memory: maps 2-bit encoded value to FP16
// encoded: 0→-1.0, 1→0.0, 2→+1.0, 3→0.0 (reserved)
__constant__ half c_lut[4];

/**
 * Main unpack kernel: W_packed [N, packed_K] → W_fp16 [N, K]
 *
 * Uses vectorized loads (uint32) to read 4 bytes at once = 16 weights.
 * Coalesced writes via half2 stores.
 */
__global__ void __launch_bounds__(BLOCK_SIZE)
int2_unpack_kernel(
    const uint8_t* __restrict__ W_packed,  // [N, packed_K]
    half* __restrict__ W_fp16,             // [N, K]
    const int N,
    const int K,
    const int packed_K
) {
    // Global thread index processes BYTES_PER_THREAD bytes = WEIGHTS_PER_THREAD weights
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_bytes = N * packed_K;

    // Each thread handles BYTES_PER_THREAD consecutive bytes
    const int byte_start = tid * BYTES_PER_THREAD;
    if (byte_start >= total_bytes) return;

    // Try vectorized 4-byte load
    if (byte_start + 3 < total_bytes) {
        // Load 4 bytes at once (16 weights)
        uint32_t packed4 = *reinterpret_cast<const uint32_t*>(&W_packed[byte_start]);

        // Calculate output position
        const int row = byte_start / packed_K;
        const int col_byte = byte_start % packed_K;
        const int col_start = col_byte * 4;

        // Unpack all 16 weights
        half* out_ptr = &W_fp16[row * K + col_start];

        // Check if entire output fits in current row
        if (col_start + WEIGHTS_PER_THREAD <= K) {
            // Fast path: all 16 weights in same row, no bounds check needed
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                uint8_t byte_val = (packed4 >> (b * 8)) & 0xFF;

                int8_t w0 = (byte_val & 0x3) - 1;
                int8_t w1 = ((byte_val >> 2) & 0x3) - 1;
                int8_t w2 = ((byte_val >> 4) & 0x3) - 1;
                int8_t w3 = ((byte_val >> 6) & 0x3) - 1;

                int offset = b * 4;
                out_ptr[offset + 0] = __float2half((float)w0);
                out_ptr[offset + 1] = __float2half((float)w1);
                out_ptr[offset + 2] = __float2half((float)w2);
                out_ptr[offset + 3] = __float2half((float)w3);
            }
        } else {
            // Slow path: handle row boundary
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                int abs_byte = byte_start + b;
                if (abs_byte >= total_bytes) break;

                int r = abs_byte / packed_K;
                int c_byte = abs_byte % packed_K;
                int c = c_byte * 4;

                uint8_t byte_val = (packed4 >> (b * 8)) & 0xFF;

                int8_t w0 = (byte_val & 0x3) - 1;
                int8_t w1 = ((byte_val >> 2) & 0x3) - 1;
                int8_t w2 = ((byte_val >> 4) & 0x3) - 1;
                int8_t w3 = ((byte_val >> 6) & 0x3) - 1;

                half* row_ptr = &W_fp16[r * K];
                if (c < K) row_ptr[c] = __float2half((float)w0);
                if (c + 1 < K) row_ptr[c + 1] = __float2half((float)w1);
                if (c + 2 < K) row_ptr[c + 2] = __float2half((float)w2);
                if (c + 3 < K) row_ptr[c + 3] = __float2half((float)w3);
            }
        }
    } else {
        // Handle tail bytes one at a time
        for (int b = 0; b < BYTES_PER_THREAD; b++) {
            int abs_byte = byte_start + b;
            if (abs_byte >= total_bytes) break;

            int r = abs_byte / packed_K;
            int c_byte = abs_byte % packed_K;
            int c = c_byte * 4;

            uint8_t byte_val = W_packed[abs_byte];

            int8_t w0 = (byte_val & 0x3) - 1;
            int8_t w1 = ((byte_val >> 2) & 0x3) - 1;
            int8_t w2 = ((byte_val >> 4) & 0x3) - 1;
            int8_t w3 = ((byte_val >> 6) & 0x3) - 1;

            half* row_ptr = &W_fp16[r * K];
            if (c < K) row_ptr[c] = __float2half((float)w0);
            if (c + 1 < K) row_ptr[c + 1] = __float2half((float)w1);
            if (c + 2 < K) row_ptr[c + 2] = __float2half((float)w2);
            if (c + 3 < K) row_ptr[c + 3] = __float2half((float)w3);
        }
    }
}

// ============================================================
// Host Interface
// ============================================================

extern "C" {

/**
 * Unpack INT2 weights to FP16
 *
 * @param W_packed  Packed weights [N, (K+3)/4] uint8
 * @param W_fp16    Output buffer [N, K] float16 (pre-allocated)
 * @param N         Number of rows (out_features)
 * @param K         Number of columns (in_features)
 * @param stream    CUDA stream
 */
void int2_unpack_to_fp16(
    const uint8_t* W_packed,
    half* W_fp16,
    int N, int K,
    cudaStream_t stream
) {
    const int packed_K = (K + 3) / 4;
    const int total_bytes = N * packed_K;
    const int total_threads = (total_bytes + BYTES_PER_THREAD - 1) / BYTES_PER_THREAD;

    dim3 block(BLOCK_SIZE);
    dim3 grid((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int2_unpack_kernel<<<grid, block, 0, stream>>>(
        W_packed, W_fp16, N, K, packed_K
    );
}

}  // extern "C"
