/**
 * INT2 Backward Kernel with Tensor Cores
 *
 * dX = dY @ W * gamma
 *
 * Uses FP16 WMMA instructions for efficient matrix multiplication.
 * W is expanded from INT2 to FP16 in shared memory before WMMA.
 *
 * Note: dW computation is handled by the hysteresis fused kernel,
 * so this kernel only computes dX.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ============================================================
// WMMA Configuration (same as forward)
// ============================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BLOCK_M 64
#define BLOCK_N 64   // This is K dimension of output dX
#define BLOCK_K 32   // This is N dimension (reduction)

#define WARPS_M 2
#define WARPS_N 2
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * WARP_SIZE)

// ============================================================
// INT2 Unpacking (same as forward)
// ============================================================

__device__ __forceinline__ void expand_int2_x4_bw(uint8_t packed, half out[4]) {
    int8_t w0 = ((packed >> 0) & 0x3) - 1;
    int8_t w1 = ((packed >> 2) & 0x3) - 1;
    int8_t w2 = ((packed >> 4) & 0x3) - 1;
    int8_t w3 = ((packed >> 6) & 0x3) - 1;

    out[0] = __float2half((float)w0);
    out[1] = __float2half((float)w1);
    out[2] = __float2half((float)w2);
    out[3] = __float2half((float)w3);
}

// ============================================================
// Main Tensor Core Backward Kernel
// ============================================================

/**
 * INT2 Backward dX with Tensor Cores
 *
 * Computes: dX[M, K] = dY[M, N] @ W[N, K] * gamma
 *
 * Matrix dimensions:
 * - dY: [M, N] (upstream gradient)
 * - W: [N, K] stored as W_packed[N, K/4] (INT2)
 * - dX: [M, K] (gradient for input)
 *
 * For the matmul dX = dY @ W:
 * - A = dY [M, N]
 * - B = W [N, K]
 * - C = dX [M, K]
 */
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
int2_backward_dx_tc_kernel(
    const half* __restrict__ dY,          // [M, N]
    const uint8_t* __restrict__ W_packed, // [N, K/4]
    half* __restrict__ dX,                // [M, K]
    const float gamma,
    const int M,
    const int N,
    const int K
) {
    // Shared memory:
    // dY tile: [BLOCK_M][BLOCK_K] where BLOCK_K is the N dimension chunk
    // W tile: [BLOCK_K][BLOCK_N] = [N_chunk][K_chunk] row-major
    __shared__ half dY_smem[BLOCK_M][BLOCK_K + 8];
    __shared__ half W_smem[BLOCK_K][BLOCK_N + 8];

    const int packed_K = (K + 3) / 4;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    // Block position in output matrix dX
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_k = blockIdx.x * BLOCK_N;  // K dimension of dX

    // Accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));
        }
    }

    // Process N dimension in tiles (reduction dimension)
    for (int n_base = 0; n_base < N; n_base += BLOCK_K) {
        // ---------- Load dY tile to shared memory ----------
        // dY_smem[m][n] = dY[block_m + m][n_base + n]
        #pragma unroll
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
            int m_local = i / BLOCK_K;
            int n_local = i % BLOCK_K;
            int m_global = block_m + m_local;
            int n_global = n_base + n_local;

            half val = __float2half(0.0f);
            if (m_global < M && n_global < N) {
                val = dY[m_global * N + n_global];
            }
            dY_smem[m_local][n_local] = val;
        }

        // ---------- Load and expand W tile to shared memory ----------
        // W is [N, K], we need W[n_base:n_base+BLOCK_K, block_k:block_k+BLOCK_N]
        // Store as W_smem[n][k] for row-major access
        #pragma unroll
        for (int i = tid; i < BLOCK_K * (BLOCK_N / 4); i += THREADS_PER_BLOCK) {
            int n_local = i / (BLOCK_N / 4);
            int byte_local = i % (BLOCK_N / 4);
            int n_global = n_base + n_local;
            int k_global = block_k + byte_local * 4;

            half expanded[4] = {__float2half(0.0f), __float2half(0.0f),
                               __float2half(0.0f), __float2half(0.0f)};

            if (n_global < N && k_global < K) {
                int byte_idx = n_global * packed_K + (k_global / 4);
                uint8_t packed_byte = W_packed[byte_idx];
                expand_int2_x4_bw(packed_byte, expanded);
            }

            int k_base_local = byte_local * 4;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (k_base_local + j < BLOCK_N && block_k + k_base_local + j < K) {
                    W_smem[n_local][k_base_local + j] = expanded[j];
                }
            }
        }

        __syncthreads();

        // ---------- WMMA Compute ----------
        // C = A @ B where A = dY [M, N], B = W [N, K], C = dX [M, K]
        // A fragment [m, n] from dY_smem
        // B fragment [n, k] from W_smem

        #pragma unroll
        for (int nn = 0; nn < BLOCK_K; nn += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

            // Load A fragments from dY_smem
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m_offset = warp_m * 32 + mi * WMMA_M;
                wmma::load_matrix_sync(a_frag[mi],
                    &dY_smem[m_offset][nn],
                    BLOCK_K + 8);
            }

            // Load B fragments from W_smem
            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                int k_offset = warp_n * 32 + ki * WMMA_N;
                wmma::load_matrix_sync(b_frag[ki],
                    &W_smem[nn][k_offset],
                    BLOCK_N + 8);
            }

            // Compute
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ki = 0; ki < 2; ki++) {
                    wmma::mma_sync(acc[mi][ki], a_frag[mi], b_frag[ki], acc[mi][ki]);
                }
            }
        }

        __syncthreads();
    }

    // ========== Apply Gamma and Store ==========
    __shared__ half out_tiles[4][WMMA_M][WMMA_N + 8];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {
            // Apply gamma scaling
            #pragma unroll
            for (int t = 0; t < acc[mi][ki].num_elements; t++) {
                float val = __half2float(acc[mi][ki].x[t]) * gamma;
                acc[mi][ki].x[t] = __float2half(val);
            }

            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_k = block_k + warp_n * 32 + ki * WMMA_N;

            wmma::store_matrix_sync(&out_tiles[warp_id][0][0], acc[mi][ki],
                WMMA_N + 8, wmma::mem_row_major);

            __syncwarp();

            for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                int m_idx = idx / WMMA_N;
                int k_idx = idx % WMMA_N;
                int m_out = out_m + m_idx;
                int k_out = out_k + k_idx;

                if (m_out < M && k_out < K) {
                    dX[m_out * K + k_out] = out_tiles[warp_id][m_idx][k_idx];
                }
            }
        }
    }
}

// ============================================================
// Simple Fallback Kernel
// ============================================================

__global__ void int2_backward_dx_tc_simple_kernel(
    const half* __restrict__ dY,
    const uint8_t* __restrict__ W_packed,
    half* __restrict__ dX,
    const float gamma,
    const int M,
    const int N,
    const int K
) {
    const int packed_K = (K + 3) / 4;

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || k >= K) return;

    float sum = 0.0f;

    for (int n = 0; n < N; n++) {
        float dy_val = __half2float(dY[m * N + n]);

        int byte_idx = k / 4;
        int bit_offset = (k % 4) * 2;
        uint8_t packed = W_packed[n * packed_K + byte_idx];
        int8_t w_val = ((packed >> bit_offset) & 0x3) - 1;

        sum += dy_val * (float)w_val;
    }

    dX[m * K + k] = __float2half(sum * gamma);
}

// ============================================================
// Host Interface
// ============================================================

extern "C" {

/**
 * INT2 backward pass dX with Tensor Cores
 *
 * Computes: dX[M, K] = dY[M, N] @ W[N, K] * gamma
 */
void int2_backward_dx_tc(
    const half* dY,
    const uint8_t* W_packed,
    half* dX,
    float gamma,
    int M, int N, int K,
    cudaStream_t stream
) {
    bool use_tc = (M >= BLOCK_M) && (K >= BLOCK_N) && (N >= BLOCK_K);

    if (use_tc) {
        dim3 grid((K + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 block(THREADS_PER_BLOCK);

        int2_backward_dx_tc_kernel<<<grid, block, 0, stream>>>(
            dY, W_packed, dX, gamma, M, N, K
        );
    } else {
        dim3 block(16, 16);
        dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        int2_backward_dx_tc_simple_kernel<<<grid, block, 0, stream>>>(
            dY, W_packed, dX, gamma, M, N, K
        );
    }
}

}  // extern "C"
