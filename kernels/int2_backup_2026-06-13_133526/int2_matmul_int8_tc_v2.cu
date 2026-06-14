/**
 * INT2 Matmul Kernel with INT8 Tensor Cores (v2 — large tiles)
 *
 * Y = X_int8 @ W_int8.T * (scale_x / 127.0) * gamma
 *
 * Tile config: 128×128×64 (vs 64×64×32 in v1)
 * Warps: 4×4 = 16 warps, 512 threads/block
 *
 * Perché tile grandi funzionano meglio:
 *  - Arithmetic intensity: ~213 FLOP/byte → compute-bound su A100 (soglia: 156 FLOP/byte)
 *  - INT8 IMMA dà 2x throughput solo se compute-bound, non se memory-bound
 *  - Tile piccole (64×64×32) erano memory-bound → nessun vantaggio da INT8
 *
 * SMEM per block (~37 KB < 48 KB limite A100):
 *  - X_smem:    int8_t  [128][80]  = 10 KB
 *  - W_smem:    int8_t  [64][144]  =  9 KB
 *  - out_tiles: int32_t [16][16][18] = 18 KB
 *
 * Occupancy: 1 block/SM (SMEM bound), ma per training con M≥2048
 * la grid è ≥512 blocks → 4+ wave su 108 SM → GPU sempre pieno.
 *
 * Target: A100 (SM 8.0+)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

#include "int2_packed.cuh"

using namespace nvcuda;

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while (0)
#endif

#ifndef CUDA_CHECK_RETURN
#define CUDA_CHECK_RETURN(call, fallback) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return (fallback); \
    } \
} while (0)
#endif

// ============================================================
// WMMA Configuration
// ============================================================

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Tile grandi: 2x rispetto alla v1 in ogni dimensione
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K  64

// 4×4 = 16 warps, 512 thread/block
#define WARPS_M 4
#define WARPS_N 4
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * WARP_SIZE)  // 512

// Tile medie: 64×64×64 — per GPU con SM < 256 block dalla v2 (es. RTX 4070 + d_model=768)
#define BLOCK_M_MED  64
#define BLOCK_N_MED  64
#define BLOCK_K_MED  64
#define WARPS_M_MED   2
#define WARPS_N_MED   2
#define THREADS_MED  (WARPS_M_MED * WARPS_N_MED * WARP_SIZE)  // 128

// Padding SMEM per alignment WMMA (stride multiplo di 16) e no bank conflict
#define SMEM_PAD_X   16   // stride X = 64+16 = 80  (multiplo di 16 ✓)
#define SMEM_PAD_W   16   // stride W = 128+16 = 144 (multiplo di 16 ✓)
#define SMEM_PAD_OUT  2   // stride out = (16+2)*4 = 72 bytes

// ============================================================
// INT2 → INT8 Unpack
// ============================================================

__device__ __forceinline__ void expand_int2_x4_int8(uint8_t packed, int8_t out[4]) {
    out[0] = bitpack::decode_int2_bits((packed >> 0) & 0x3);
    out[1] = bitpack::decode_int2_bits((packed >> 2) & 0x3);
    out[2] = bitpack::decode_int2_bits((packed >> 4) & 0x3);
    out[3] = bitpack::decode_int2_bits((packed >> 6) & 0x3);
}

// ============================================================
// Kernel principale
// ============================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
int2_matmul_int8_tc_v2_kernel(
    const int8_t*  __restrict__ X_int8,    // [M, K]
    const uint8_t* __restrict__ W_packed,  // [N, K/4]
    half*          __restrict__ Y,         // [M, N]
    const float*   __restrict__ d_scale_x,
    const float    gamma,
    const int M, const int N, const int K
) {
    // ---- SMEM ----
    __shared__ int8_t  X_smem[BLOCK_M][BLOCK_K + SMEM_PAD_X];
    __shared__ int8_t  W_smem[BLOCK_K][BLOCK_N + SMEM_PAD_W];
    __shared__ int32_t out_tiles[WARPS_M * WARPS_N][WMMA_M][WMMA_N + SMEM_PAD_OUT];

    // Out scale: un solo accesso globale per block
    __shared__ float s_out_scale;
    if (threadIdx.x == 0)
        s_out_scale = (*d_scale_x / 127.0f) * gamma;
    __syncthreads();
    const float out_scale = s_out_scale;

    const int packed_K = (K + 3) / 4;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int warp_m   = warp_id / WARPS_N;   // [0,3]
    const int warp_n   = warp_id % WARPS_N;   // [0,3]

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_n = blockIdx.x * BLOCK_N;

    // ---- Accumulatori INT32 ----
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    // ---- K loop ----
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {

        // Carica X_int8: BLOCK_M * BLOCK_K = 128*64 = 8192 elem, 512 thread → 16 elem/thread
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * BLOCK_K; i += THREADS_PER_BLOCK) {
            int m_l = i / BLOCK_K, k_l = i % BLOCK_K;
            int m_g = block_m + m_l, k_g = k_base + k_l;
            X_smem[m_l][k_l] = (m_g < M && k_g < K) ? X_int8[m_g * K + k_g] : (int8_t)0;
        }

        // Espandi W: BLOCK_N * (BLOCK_K/4) = 128*16 = 2048 byte → 4 iter/thread
        #pragma unroll 2
        for (int i = tid; i < BLOCK_N * (BLOCK_K / 4); i += THREADS_PER_BLOCK) {
            int n_l    = i / (BLOCK_K / 4);
            int byte_l = i % (BLOCK_K / 4);
            int n_g    = block_n + n_l;
            int k_g    = k_base + byte_l * 4;

            int8_t exp[4] = {0, 0, 0, 0};
            if (n_g < N && k_g < K)
                expand_int2_x4_int8(W_packed[n_g * packed_K + (k_g / 4)], exp);

            int k_bl = byte_l * 4;
            #pragma unroll
            for (int j = 0; j < 4; j++)
                if (k_bl + j < BLOCK_K && k_base + k_bl + j < K)
                    W_smem[k_bl + j][n_l] = exp[j];
        }

        __syncthreads();

        // WMMA: ogni warp fa 2×2 = 4 tile output (32×32)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           int8_t, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           int8_t, wmma::row_major> b_frag[2];

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi],
                    &X_smem[warp_m * 32 + mi * WMMA_M][kk],
                    BLOCK_K + SMEM_PAD_X);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(b_frag[ni],
                    &W_smem[kk][warp_n * 32 + ni * WMMA_N],
                    BLOCK_N + SMEM_PAD_W);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }

        __syncthreads();
    }

    // ---- Store: INT32 * scale → FP16 → global ----
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_n = block_n + warp_n * 32 + ni * WMMA_N;

            wmma::store_matrix_sync(
                &out_tiles[warp_id][0][0], acc[mi][ni],
                WMMA_N + SMEM_PAD_OUT, wmma::mem_row_major);
            __syncwarp();

            for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                int m_i = idx / WMMA_N, n_i = idx % WMMA_N;
                int m_o = out_m + m_i, n_o = out_n + n_i;
                if (m_o < M && n_o < N)
                    Y[m_o * N + n_o] = __float2half(
                        (float)out_tiles[warp_id][m_i][n_i] * out_scale);
            }
        }
    }
}

// ============================================================
// Medium-tile kernel: 64×64×64, 4 warp, 128 thread/block
// SMEM ~14.8 KB → 6 block/SM su RTX 4070 (100 KB) → tutti gli SM attivi
// ============================================================

__global__ void __launch_bounds__(THREADS_MED)
int2_matmul_int8_tc_v2_medium_kernel(
    const int8_t*  __restrict__ X_int8,
    const uint8_t* __restrict__ W_packed,
    half*          __restrict__ Y,
    const float*   __restrict__ d_scale_x,
    const float    gamma,
    const int M, const int N, const int K
) {
    __shared__ int8_t  X_smem[BLOCK_M_MED][BLOCK_K_MED + SMEM_PAD_X];
    __shared__ int8_t  W_smem[BLOCK_K_MED][BLOCK_N_MED + SMEM_PAD_W];
    __shared__ int32_t out_tiles[WARPS_M_MED * WARPS_N_MED][WMMA_M][WMMA_N + SMEM_PAD_OUT];

    __shared__ float s_out_scale;
    if (threadIdx.x == 0)
        s_out_scale = (*d_scale_x / 127.0f) * gamma;
    __syncthreads();
    const float out_scale = s_out_scale;

    const int packed_K = (K + 3) / 4;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int warp_m   = warp_id / WARPS_N_MED;   // [0,1]
    const int warp_n   = warp_id % WARPS_N_MED;   // [0,1]

    const int block_m = blockIdx.y * BLOCK_M_MED;
    const int block_n = blockIdx.x * BLOCK_N_MED;

    // Accumulatori INT32: 2×2 tile per warp (32×32 output, identico a v2)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    for (int k_base = 0; k_base < K; k_base += BLOCK_K_MED) {

        // Carica X_int8: 64*64 = 4096 elem, 128 thread → 32 elem/thread
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M_MED * BLOCK_K_MED; i += THREADS_MED) {
            int m_l = i / BLOCK_K_MED, k_l = i % BLOCK_K_MED;
            int m_g = block_m + m_l, k_g = k_base + k_l;
            X_smem[m_l][k_l] = (m_g < M && k_g < K) ? X_int8[m_g * K + k_g] : (int8_t)0;
        }

        // Espandi W: 64*(64/4) = 1024 byte → 8 iter/thread
        #pragma unroll 2
        for (int i = tid; i < BLOCK_N_MED * (BLOCK_K_MED / 4); i += THREADS_MED) {
            int n_l    = i / (BLOCK_K_MED / 4);
            int byte_l = i % (BLOCK_K_MED / 4);
            int n_g    = block_n + n_l;
            int k_g    = k_base + byte_l * 4;

            int8_t exp[4] = {0, 0, 0, 0};
            if (n_g < N && k_g < K)
                expand_int2_x4_int8(W_packed[n_g * packed_K + (k_g / 4)], exp);

            int k_bl = byte_l * 4;
            #pragma unroll
            for (int j = 0; j < 4; j++)
                if (k_bl + j < BLOCK_K_MED && k_base + k_bl + j < K)
                    W_smem[k_bl + j][n_l] = exp[j];
        }

        __syncthreads();

        // WMMA: ogni warp calcola 2×2 tile = 32×32 output (BLOCK dimezzato vs v2, stessa struttura)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K_MED; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag[2];

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi],
                    &X_smem[warp_m * 32 + mi * WMMA_M][kk],
                    BLOCK_K_MED + SMEM_PAD_X);

            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(b_frag[ni],
                    &W_smem[kk][warp_n * 32 + ni * WMMA_N],
                    BLOCK_N_MED + SMEM_PAD_W);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < 2; ni++)
                    wmma::mma_sync(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
        }

        __syncthreads();
    }

    // Store: INT32 * scale → FP16 → global
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_n = block_n + warp_n * 32 + ni * WMMA_N;

            wmma::store_matrix_sync(
                &out_tiles[warp_id][0][0], acc[mi][ni],
                WMMA_N + SMEM_PAD_OUT, wmma::mem_row_major);
            __syncwarp();

            for (int idx = lane_id; idx < WMMA_M * WMMA_N; idx += WARP_SIZE) {
                int m_i = idx / WMMA_N, n_i = idx % WMMA_N;
                int m_o = out_m + m_i, n_o = out_n + n_i;
                if (m_o < M && n_o < N)
                    Y[m_o * N + n_o] = __float2half(
                        (float)out_tiles[warp_id][m_i][n_i] * out_scale);
            }
        }
    }
}

// ============================================================
// Fallback scalar (matrici piccole < BLOCK_M/BLOCK_N)
// ============================================================

__global__ void int2_matmul_int8_tc_v2_simple_kernel(
    const int8_t*  __restrict__ X_int8,
    const uint8_t* __restrict__ W_packed,
    half*          __restrict__ Y,
    const float*   __restrict__ d_scale_x,
    const float gamma,
    const int M, const int N, const int K
) {
    const float out_scale = (*d_scale_x / 127.0f) * gamma;
    const int packed_K = (K + 3) / 4;

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || n >= N) return;

    int32_t sum = 0;
    for (int k = 0; k < K; k++) {
        int8_t w = bitpack::decode_int2_bits((W_packed[n * packed_K + k/4] >> ((k%4)*2)) & 0x3);
        sum += (int32_t)X_int8[m * K + k] * (int32_t)w;
    }
    Y[m * N + n] = __float2half((float)sum * out_scale);
}

// ============================================================
// Host interface
// ============================================================

// ============================================================
// Device properties cache (lazy init, costo zero dopo prima chiamata)
// ============================================================

static int s_fwd_device = -1;
static int s_fwd_num_sms = -1;

static inline int _fwd_get_num_sms() {
    int device = 0;
    CUDA_CHECK_RETURN(cudaGetDevice(&device), 1);
    if (s_fwd_num_sms >= 0 && s_fwd_device == device) return s_fwd_num_sms;

    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device), 1);
    s_fwd_device = device;
    s_fwd_num_sms = prop.multiProcessorCount;
    return s_fwd_num_sms;
}

extern "C" {

void int2_matmul_int8_tc_v2(
    const int8_t*  X_int8,
    const uint8_t* W_packed,
    half*          Y,
    const float*   d_scale_x,
    float          gamma,
    int M, int N, int K,
    cudaStream_t   stream
) {
    int num_sms = _fwd_get_num_sms();

    // Blocchi che produrrebbe ciascun kernel
    int blocks_large = ((N + BLOCK_N - 1) / BLOCK_N) * ((M + BLOCK_M - 1) / BLOCK_M);
    int blocks_med   = ((N + BLOCK_N_MED - 1) / BLOCK_N_MED) * ((M + BLOCK_M_MED - 1) / BLOCK_M_MED);

    if (blocks_large >= num_sms * 2 && M >= BLOCK_M && N >= BLOCK_N && K >= BLOCK_K) {
        // Kernel v2 originale: grandi batch / grande d_model (A100, H100)
        dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        int2_matmul_int8_tc_v2_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            X_int8, W_packed, Y, d_scale_x, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());

    } else if (blocks_med >= num_sms && M >= BLOCK_M_MED && N >= BLOCK_N_MED && K >= BLOCK_K_MED) {
        // Kernel medium: satura tutti gli SM su GPU con d_model piccolo (es. RTX 4070 + 768)
        dim3 grid((N + BLOCK_N_MED - 1) / BLOCK_N_MED, (M + BLOCK_M_MED - 1) / BLOCK_M_MED);
        int2_matmul_int8_tc_v2_medium_kernel<<<grid, THREADS_MED, 0, stream>>>(
            X_int8, W_packed, Y, d_scale_x, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());

    } else {
        // Fallback scalar per matrici molto piccole
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        int2_matmul_int8_tc_v2_simple_kernel<<<grid, block, 0, stream>>>(
            X_int8, W_packed, Y, d_scale_x, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // extern "C"
