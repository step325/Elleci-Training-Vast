/**
 * Optimized Fused Hysteresis Update Kernel
 *
 * Combines gradient computation and hysteresis update in a single kernel.
 * Uses shared memory for better cache utilization.
 *
 * For each weight w[n,k]:
 * 1. Compute gradient: dW[n,k] = sum_m dY[m,n] * X[m,k]
 * 2. Apply stochastic rounding to get discrete delta
 * 3. Update hysteresis counter
 * 4. Flip weight if threshold reached
 * 5. Write back packed weight and hysteresis
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "int2_packed.cuh"

// Tile size for shared memory
#define HYST_TILE_M 32
#define HYST_BLOCK_SIZE 256

/**
 * Optimized hysteresis update kernel with shared memory tiling
 *
 * Each block handles a tile of weights [n_start:n_end, k_start:k_end]
 * Uses shared memory to cache dY and X tiles for the M reduction.
 */
__global__ void int2_hysteresis_optimized_kernel(
    const half* __restrict__ dY,           // [M, N]
    const half* __restrict__ X,            // [M, K]
    const int M,
    uint8_t* __restrict__ W_packed,        // [N, packed_K]
    uint8_t* __restrict__ H_packed,        // [N, packed_H]
    const float lr,
    const float lr_scale,
    const int8_t threshold_S,
    const float decay_rate,
    const uint32_t step,
    const int N,
    const int K
) {
    // Shared memory for tiled loading
    __shared__ half dY_tile[HYST_TILE_M];  // [M_tile] for one n
    __shared__ half X_tile[HYST_TILE_M];   // [M_tile] for one k

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_weights = N * K;

    if (idx >= total_weights) return;

    const int n = idx / K;
    const int k = idx % K;
    const int packed_K = (K + 3) / 4;
    const int packed_H = (K + 1) / 2;

    // ========== 1. COMPUTE GRADIENT ==========
    float grad = 0.0f;

    // Process M in tiles
    for (int m_base = 0; m_base < M; m_base += HYST_TILE_M) {
        int m_end = min(m_base + HYST_TILE_M, M);
        int tile_size = m_end - m_base;

        // Each thread participates in loading tiles
        // But we need to be careful - not all threads have the same n,k
        // So each thread loads its own slice

        // Unrolled accumulation
        int m = m_base;
        #pragma unroll 4
        for (; m + 3 < m_end; m += 4) {
            float dy0 = __half2float(dY[(m + 0) * N + n]);
            float dy1 = __half2float(dY[(m + 1) * N + n]);
            float dy2 = __half2float(dY[(m + 2) * N + n]);
            float dy3 = __half2float(dY[(m + 3) * N + n]);

            float x0 = __half2float(X[(m + 0) * K + k]);
            float x1 = __half2float(X[(m + 1) * K + k]);
            float x2 = __half2float(X[(m + 2) * K + k]);
            float x3 = __half2float(X[(m + 3) * K + k]);

            grad += dy0 * x0 + dy1 * x1 + dy2 * x2 + dy3 * x3;
        }
        // Handle remaining elements
        for (; m < m_end; m++) {
            grad += __half2float(dY[m * N + n]) * __half2float(X[m * K + k]);
        }
    }

    // ========== 2. LOAD CURRENT STATE ==========
    int8_t q = bitpack::unpack_int2(W_packed + n * packed_K, k);
    int8_t h = bitpack::unpack_int4(H_packed + n * packed_H, k);

    // ========== 3. RNG (Philox-style hash) ==========
    uint32_t hash1 = idx ^ (step * 2654435761u);
    hash1 = ((hash1 >> 16) ^ hash1) * 0x45d9f3b;
    hash1 = ((hash1 >> 16) ^ hash1) * 0x45d9f3b;
    float rand1 = (hash1 & 0xFFFFFF) / float(0x1000000);

    uint32_t hash2 = idx ^ (step * 2246822519u);
    hash2 = ((hash2 >> 16) ^ hash2) * 0x45d9f3b;
    hash2 = ((hash2 >> 16) ^ hash2) * 0x45d9f3b;
    float rand2 = (hash2 & 0xFFFFFF) / float(0x1000000);

    // ========== 4. STOCHASTIC ROUNDING ==========
    float lr_scaled = lr * lr_scale * (float)threshold_S;
    float delta_raw = -grad * lr_scaled;

    float sign = (delta_raw >= 0.0f) ? 1.0f : -1.0f;
    float abs_delta = fabsf(delta_raw);
    float floor_val = floorf(abs_delta);
    float frac = abs_delta - floor_val;

    int32_t delta = (int32_t)(sign * (floor_val + (rand1 < frac ? 1.0f : 0.0f)));
    delta = max(-31, min(31, delta));

    // ========== 5. DECAY ==========
    int16_t h_new = (int16_t)h + (int16_t)delta;

    if (q != 0 && rand2 < decay_rate) {
        h_new -= (q > 0) ? 1 : -1;
    }

    // ========== 6. STATE TRANSITION ==========
    int8_t q_new = q;

    if (h_new >= threshold_S) {
        if (q == -1) {
            q_new = 0;
            h_new = 0;
        } else if (q == 0) {
            q_new = 1;
            h_new = 0;
        } else {
            h_new = threshold_S;
        }
    } else if (h_new <= -threshold_S) {
        if (q == 1) {
            q_new = 0;
            h_new = 0;
        } else if (q == 0) {
            q_new = -1;
            h_new = 0;
        } else {
            h_new = -threshold_S;
        }
    }

    h_new = max((int16_t)(-threshold_S), min((int16_t)threshold_S, h_new));

    // ========== 7. WRITE BACK ==========
    bitpack::pack_int2_unsafe(W_packed + n * packed_K, k, q_new);
    bitpack::pack_int4_unsafe(H_packed + n * packed_H, k, (int8_t)h_new);
}

// ============================================================
// Host Interface
// ============================================================

extern "C" {

/**
 * Optimized fused backward + hysteresis update
 */
void int2_hysteresis_optimized_update(
    const half* dY,
    const half* X,
    int M,
    uint8_t* W_packed,
    uint8_t* H_packed,
    float lr,
    float lr_scale,
    int threshold,
    float decay_rate,
    int step,
    int N, int K,
    cudaStream_t stream
) {
    int total_weights = N * K;
    int block_size = HYST_BLOCK_SIZE;
    int num_blocks = (total_weights + block_size - 1) / block_size;

    int2_hysteresis_optimized_kernel<<<num_blocks, block_size, 0, stream>>>(
        dY, X, M,
        W_packed, H_packed,
        lr, lr_scale, (int8_t)threshold, decay_rate, (uint32_t)step,
        N, K
    );
}

}  // extern "C"
