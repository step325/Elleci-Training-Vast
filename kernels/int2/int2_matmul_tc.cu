/**
 * INT2 Matmul Kernel with Tensor Cores
 *
 * Y = X @ W.T * gamma
 *
 * Uses FP16 WMMA instructions for efficient matrix multiplication.
 * W is expanded from INT2 to FP16 in shared memory before WMMA.
 *
 * Target: NVIDIA Ada Lovelace (SM 8.9) - RTX 4070/4080/4090
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

// ============================================================
// WMMA Configuration
// ============================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block-level tiles
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32

// Each block has 4 warps (128 threads)
// 2 warps in M direction, 2 in N direction
// Each warp computes a 32x32 tile (2x2 WMMA tiles)
#define WARPS_M 2
#define WARPS_N 2
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * WARP_SIZE)

// ============================================================
// INT2 Unpacking Utilities
// ============================================================

/**
 * Expand INT2 packed byte to 4 FP16 values
 * INT2 encoding: 00 = -1, 01 = 0, 10 = +1, 11 = reserved (treated as 0)
 */
__device__ __forceinline__ void expand_int2_x4(uint8_t packed, half out[4]) {
    // Decode each 2-bit value: encoded - 1 gives the actual weight
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
// Main Tensor Core Kernel
// ============================================================

/**
 * INT2 Matmul with Tensor Cores
 *
 * Computes: Y[M, N] = X[M, K] @ W[N, K].T * gamma
 *
 * Block: 128 threads = 4 warps
 * Grid: ceil(N/BLOCK_N) x ceil(M/BLOCK_M)
 *
 * Each block computes BLOCK_M x BLOCK_N output tile.
 * Each warp computes 32x32 = 4 WMMA tiles of 16x16.
 */
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
int2_matmul_tc_kernel(
    const half* __restrict__ X,           // [M, K]
    const uint8_t* __restrict__ W_packed, // [N, K/4]
    half* __restrict__ Y,                 // [M, N]
    const float gamma,
    const int M,
    const int N,
    const int K
) {
    // ========== Shared Memory ==========
    // X tile: [BLOCK_M][BLOCK_K] in row-major
    // W tile: [BLOCK_K][BLOCK_N] in col-major (transposed for efficient access)
    __shared__ half X_smem[BLOCK_M][BLOCK_K + 8];  // +8 to avoid bank conflicts
    __shared__ half W_smem[BLOCK_K][BLOCK_N + 8];  // Stored as W.T layout

    const int packed_K = (K + 3) / 4;

    // ========== Thread/Warp Indices ==========
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Warp position in block tile
    const int warp_m = warp_id / WARPS_N;  // 0 or 1
    const int warp_n = warp_id % WARPS_N;  // 0 or 1

    // Block position in output matrix
    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    // ========== Accumulator Fragments ==========
    // Each warp computes 2x2 WMMA tiles = 32x32 output
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc[2][2];

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));
        }
    }

    // ========== Main K Loop ==========
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        // ---------- Load X tile to shared memory ----------
        // Each thread loads multiple elements
        // X_smem[m][k] = X[block_m + m][k_base + k]
        #pragma unroll
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
            int m_local = i / BLOCK_K;
            int k_local = i % BLOCK_K;
            int m_global = block_m + m_local;
            int k_global = k_base + k_local;

            half val = __float2half(0.0f);
            if (m_global < M && k_global < K) {
                val = X[m_global * K + k_global];
            }
            X_smem[m_local][k_local] = val;
        }

        // ---------- Load and expand W tile to shared memory ----------
        // W is stored as [N, K] but we want W.T for matmul
        // W_smem[k][n] = W[block_n + n][k_base + k] (expanded from INT2)
        //
        // Each byte in W_packed contains 4 weights (INT2)
        // We need to load BLOCK_N * BLOCK_K weights = BLOCK_N * BLOCK_K/4 bytes

        #pragma unroll
        for (int i = tid; i < BLOCK_N * (BLOCK_K / 4); i += THREADS_PER_BLOCK) {
            int n_local = i / (BLOCK_K / 4);
            int byte_local = i % (BLOCK_K / 4);
            int n_global = block_n + n_local;
            int k_global = k_base + byte_local * 4;

            half expanded[4] = {__float2half(0.0f), __float2half(0.0f),
                               __float2half(0.0f), __float2half(0.0f)};

            if (n_global < N && k_global < K) {
                // Calculate byte index in W_packed
                int byte_idx = n_global * packed_K + (k_global / 4);
                uint8_t packed_byte = W_packed[byte_idx];
                expand_int2_x4(packed_byte, expanded);
            }

            // Store expanded weights in transposed layout
            // W_smem[k][n] for k in [byte_local*4, byte_local*4+4)
            int k_base_local = byte_local * 4;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (k_base_local + j < BLOCK_K && k_base + k_base_local + j < K) {
                    W_smem[k_base_local + j][n_local] = expanded[j];
                }
            }
        }

        __syncthreads();

        // ---------- WMMA Compute ----------
        // Process BLOCK_K in WMMA_K chunks
        //
        // Matrix multiply: C = A @ B where:
        //   A = X [M, K] stored in X_smem[M][K] row-major
        //   B = W.T [K, N] stored in W_smem[K][N] row-major
        //   C = Y [M, N]
        //
        // For WMMA with row_major layouts:
        //   A fragment [m, k]: A[i,j] = X_smem[m_start + i][k_start + j]
        //   B fragment [k, n]: B[i,j] = W_smem[k_start + i][n_start + j]

        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            // Load fragments for this warp
            // Both A and B use row_major since our storage is row-major
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];

            // Each warp handles 32x32 output = 2x2 WMMA tiles
            // Warp (warp_m, warp_n) handles:
            //   M: [warp_m*32, warp_m*32+32)
            //   N: [warp_n*32, warp_n*32+32)

            // Load A fragments (from X_smem)
            // X_smem is [BLOCK_M][BLOCK_K+8], leading dim = BLOCK_K + 8
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int m_offset = warp_m * 32 + mi * WMMA_M;
                wmma::load_matrix_sync(a_frag[mi],
                    &X_smem[m_offset][kk],
                    BLOCK_K + 8);  // Leading dimension
            }

            // Load B fragments (from W_smem)
            // W_smem is [BLOCK_K][BLOCK_N+8], leading dim = BLOCK_N + 8
            #pragma unroll
            for (int ni = 0; ni < 2; ni++) {
                int n_offset = warp_n * 32 + ni * WMMA_N;
                wmma::load_matrix_sync(b_frag[ni],
                    &W_smem[kk][n_offset],
                    BLOCK_N + 8);  // Leading dimension
            }

            // Compute all 2x2 = 4 output tiles
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // ========== Apply Gamma and Store ==========
    // Use separate output buffer per warp to avoid race conditions
    // 4 warps × 16×24 (padded) = 1536 half = 3 KB shared memory
    __shared__ half out_tiles[4][WMMA_M][WMMA_N + 8];

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            // Apply gamma scaling to accumulator
            #pragma unroll
            for (int t = 0; t < acc[mi][ni].num_elements; t++) {
                float val = __half2float(acc[mi][ni].x[t]) * gamma;
                acc[mi][ni].x[t] = __float2half(val);
            }

            // Calculate output position
            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_n = block_n + warp_n * 32 + ni * WMMA_N;

            // Store to warp-local shared memory buffer
            wmma::store_matrix_sync(&out_tiles[warp_id][0][0], acc[mi][ni],
                WMMA_N + 8, wmma::mem_row_major);

            __syncwarp();

            // Copy to global with bounds checking
            // Each lane handles part of the 16x16 tile
            for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                int m_idx = idx / WMMA_N;
                int n_idx = idx % WMMA_N;
                int m_out = out_m + m_idx;
                int n_out = out_n + n_idx;

                if (m_out < M && n_out < N) {
                    Y[m_out * N + n_out] = out_tiles[warp_id][m_idx][n_idx];
                }
            }
        }
    }
}

// ============================================================
// Simple Fallback Kernel (for small matrices)
// ============================================================

__global__ void int2_matmul_tc_simple_kernel(
    const half* __restrict__ X,
    const uint8_t* __restrict__ W_packed,
    half* __restrict__ Y,
    const float gamma,
    const int M,
    const int N,
    const int K
) {
    const int packed_K = (K + 3) / 4;

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (m >= M || n >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        float x_val = __half2float(X[m * K + k]);

        // Unpack weight W[n, k]
        int byte_idx = k / 4;
        int bit_offset = (k % 4) * 2;
        uint8_t packed = W_packed[n * packed_K + byte_idx];
        int8_t w_val = ((packed >> bit_offset) & 0x3) - 1;

        sum += x_val * (float)w_val;
    }

    Y[m * N + n] = __float2half(sum * gamma);
}

// ============================================================
// Host Interface
// ============================================================

extern "C" {

/**
 * INT2 matmul forward pass with Tensor Cores
 *
 * @param X Input tensor [M, K] float16
 * @param W_packed Packed weights [N, (K+3)/4] uint8
 * @param Y Output tensor [M, N] float16
 * @param gamma Scale factor
 * @param M Batch * sequence length
 * @param N Output features
 * @param K Input features
 * @param stream CUDA stream
 */
void int2_matmul_tc(
    const half* X,
    const uint8_t* W_packed,
    half* Y,
    float gamma,
    int M, int N, int K,
    cudaStream_t stream
) {
    // Check if we can use the TC kernel
    bool use_tc = (M >= BLOCK_M) && (N >= BLOCK_N) && (K >= BLOCK_K);

    if (use_tc) {
        // Use Tensor Core kernel
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        dim3 block(THREADS_PER_BLOCK);

        int2_matmul_tc_kernel<<<grid, block, 0, stream>>>(
            X, W_packed, Y, gamma, M, N, K
        );
    } else {
        // Use simple kernel for small matrices
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

        int2_matmul_tc_simple_kernel<<<grid, block, 0, stream>>>(
            X, W_packed, Y, gamma, M, N, K
        );
    }
}

}  // extern "C"
