/**
 * INT2 Backward dX con INT8 Tensor Cores (v2 — large tiles)
 *
 * dX = dY_int8 @ W_int8 * (scale_dy / 127.0) * gamma
 *
 * Stesso layout tile del forward v2: 128×128×64, 16 warps, 512 thread/block.
 * SMEM: ~37 KB < 48 KB limite A100.
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

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// BLOCK_M: tile batch (M), BLOCK_N: tile output K, BLOCK_K: riduzione N
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K  64

#define WARPS_M 4
#define WARPS_N 4
#define WARP_SIZE 32
#define THREADS_PER_BLOCK (WARPS_M * WARPS_N * WARP_SIZE)  // 512

// Tile medie: 64×64×64 — stessa logica del forward
#define BLOCK_M_MED  64
#define BLOCK_N_MED  64
#define BLOCK_K_MED  64
#define WARPS_M_MED   2
#define WARPS_N_MED   2
#define THREADS_MED  (WARPS_M_MED * WARPS_N_MED * WARP_SIZE)  // 128

#define SMEM_PAD_DY  16
#define SMEM_PAD_W   16
#define SMEM_PAD_OUT  4

__device__ __forceinline__ void expand_int2_x4_int8_bw(uint8_t packed, int8_t out[4]) {
    out[0] = bitpack::decode_int2_bits((packed >> 0) & 0x3);
    out[1] = bitpack::decode_int2_bits((packed >> 2) & 0x3);
    out[2] = bitpack::decode_int2_bits((packed >> 4) & 0x3);
    out[3] = bitpack::decode_int2_bits((packed >> 6) & 0x3);
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
int2_backward_dx_int8_tc_v2_kernel(
    const int8_t*  __restrict__ dY_int8,    // [M, N]
    const uint8_t* __restrict__ W_packed,   // [N, K/4]
    half*          __restrict__ dX,         // [M, K]
    const float*   __restrict__ d_scale_dy,
    const float    gamma,
    const int M, const int N, const int K
) {
    __shared__ int8_t  dY_smem[BLOCK_M][BLOCK_K + SMEM_PAD_DY];
    __shared__ int8_t  W_smem[BLOCK_K][BLOCK_N + SMEM_PAD_W];
    __shared__ int32_t out_tiles[WARPS_M * WARPS_N][WMMA_M][WMMA_N + SMEM_PAD_OUT];

    __shared__ float s_out_scale;
    if (threadIdx.x == 0)
        s_out_scale = (*d_scale_dy / 127.0f) * gamma;
    __syncthreads();
    const float out_scale = s_out_scale;

    const int packed_K_dim = (K + 3) / 4;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m  = warp_id / WARPS_N;
    const int warp_n  = warp_id % WARPS_N;

    const int block_m = blockIdx.y * BLOCK_M;
    const int block_k = blockIdx.x * BLOCK_N;  // K-output tile

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    // N loop (riduzione)
    for (int n_base = 0; n_base < N; n_base += BLOCK_K) {

        // Carica dY_int8 vettorizzato (16 bytes = 16 int8 per thread)
        #pragma unroll
        for (int i = tid; i < (BLOCK_M * BLOCK_K) / 16; i += THREADS_PER_BLOCK) {
            int m_l = (i * 16) / BLOCK_K;
            int n_l = (i * 16) % BLOCK_K;
            int m_g = block_m + m_l;
            int n_g = n_base + n_l;

            int4 val = {0, 0, 0, 0};
            // int4 (16B) richiede N%16==0 (n_g è multiplo di 16); altrimenti scalare.
            if ((N % 16 == 0) && m_g < M && n_g + 15 < N) {
                val = *reinterpret_cast<const int4*>(&dY_int8[m_g * N + n_g]);
            } else {
                __align__(16) int8_t tmp[16] = {0};
                for(int j=0; j<16; j++) {
                    if (m_g < M && n_g + j < N) tmp[j] = dY_int8[m_g * N + n_g + j];
                }
                val = *reinterpret_cast<int4*>(&tmp[0]);
            }
            *reinterpret_cast<int4*>(&dY_smem[m_l][n_l]) = val;
        }

        // Espandi W_packed vettorizzato (4 bytes = 16 pesi per thread)
        #pragma unroll
        for (int i = tid; i < (BLOCK_K * BLOCK_N) / 16; i += THREADS_PER_BLOCK) {
            int n_l    = i / (BLOCK_N / 16);
            int byte_chunk_l = i % (BLOCK_N / 16);
            int n_g    = n_base + n_l;
            int byte_g_start = (block_k / 4) + byte_chunk_l * 4;

            uint32_t packed_val = 0;
            // uint32 (4B) richiede packed_K_dim%4==0 (byte_g_start è multiplo di 4).
            if ((packed_K_dim % 4 == 0) && n_g < N && byte_g_start * 4 + 15 < K) {
                packed_val = *reinterpret_cast<const uint32_t*>(&W_packed[n_g * packed_K_dim + byte_g_start]);
            } else {
                __align__(4) uint8_t tmp[4] = {0};
                for(int j=0; j<4; j++) {
                    if (n_g < N && (byte_g_start + j)*4 < K) 
                        tmp[j] = W_packed[n_g * packed_K_dim + byte_g_start + j];
                }
                packed_val = *reinterpret_cast<uint32_t*>(&tmp[0]);
            }

            uint8_t* bytes = (uint8_t*)&packed_val;
            
            #pragma unroll
            for(int b=0; b<4; b++) {
                __align__(4) int8_t exp[4];  // letto via reinterpret_cast<int*>
                expand_int2_x4_int8_bw(bytes[b], exp);
                int k_bl = (byte_chunk_l * 4 + b) * 4;
                if (k_bl + 3 < BLOCK_N) {
                    *reinterpret_cast<int*>(&W_smem[n_l][k_bl]) = *reinterpret_cast<int*>(&exp[0]);
                } else {
                    #pragma unroll
                    for (int j=0; j<4; j++) {
                        if (k_bl + j < BLOCK_N) W_smem[n_l][k_bl + j] = exp[j];
                    }
                }
            }
        }

        __syncthreads();

        // WMMA INT8×INT8→INT32
        #pragma unroll
        for (int nn = 0; nn < BLOCK_K; nn += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           int8_t, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           int8_t, wmma::row_major> b_frag[2];

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi],
                    &dY_smem[warp_m * 32 + mi * WMMA_M][nn],
                    BLOCK_K + SMEM_PAD_DY);

            #pragma unroll
            for (int ki = 0; ki < 2; ki++)
                wmma::load_matrix_sync(b_frag[ki],
                    &W_smem[nn][warp_n * 32 + ki * WMMA_N],
                    BLOCK_N + SMEM_PAD_W);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ki = 0; ki < 2; ki++)
                    wmma::mma_sync(acc[mi][ki], a_frag[mi], b_frag[ki], acc[mi][ki]);
        }

        __syncthreads();
    }

    // Store: INT32 * scale → FP16 → global
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {
            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_k = block_k + warp_n * 32 + ki * WMMA_N;

            wmma::store_matrix_sync(
                &out_tiles[warp_id][0][0], acc[mi][ki],
                WMMA_N + SMEM_PAD_OUT, wmma::mem_row_major);
            __syncwarp();

            int row = lane_id / 2;
            int col_chunk = lane_id % 2;
            int m_o = out_m + row;
            int k_o = out_k + col_chunk * 8;

            // float4 (16B) su dX richiede K%8==0 (k_o è già multiplo di 8).
            if ((K % 8 == 0) && m_o < M && k_o + 7 < K) {
                half out_vec[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    out_vec[i] = __float2half((float)out_tiles[warp_id][row][col_chunk * 8 + i] * out_scale);
                }
                *reinterpret_cast<float4*>(&dX[m_o * K + k_o]) = *reinterpret_cast<float4*>(&out_vec[0]);
            } else {
                for (int i = 0; i < 8; i++) {
                    if (m_o < M && k_o + i < K) {
                        dX[m_o * K + k_o + i] = __float2half((float)out_tiles[warp_id][row][col_chunk * 8 + i] * out_scale);
                    }
                }
            }
        }
    }
}

// ============================================================
// Medium-tile backward: 64×64×64, 4 warp, 128 thread/block
// ============================================================

__global__ void __launch_bounds__(THREADS_MED)
int2_backward_dx_int8_tc_v2_medium_kernel(
    const int8_t*  __restrict__ dY_int8,
    const uint8_t* __restrict__ W_packed,
    half*          __restrict__ dX,
    const float*   __restrict__ d_scale_dy,
    const float    gamma,
    const int M, const int N, const int K
) {
    __shared__ int8_t  dY_smem[BLOCK_M_MED][BLOCK_K_MED + SMEM_PAD_DY];
    __shared__ int8_t  W_smem[BLOCK_K_MED][BLOCK_N_MED + SMEM_PAD_W];
    __shared__ int32_t out_tiles[WARPS_M_MED * WARPS_N_MED][WMMA_M][WMMA_N + SMEM_PAD_OUT];

    __shared__ float s_out_scale;
    if (threadIdx.x == 0)
        s_out_scale = (*d_scale_dy / 127.0f) * gamma;
    __syncthreads();
    const float out_scale = s_out_scale;

    const int packed_K_dim = (K + 3) / 4;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warp_m  = warp_id / WARPS_N_MED;
    const int warp_n  = warp_id % WARPS_N_MED;

    const int block_m = blockIdx.y * BLOCK_M_MED;
    const int block_k = blockIdx.x * BLOCK_N_MED;

    // Accumulatori INT32: 2×2 tile per warp (32×32 output, identico a v2)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc[2][2];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0);

    for (int n_base = 0; n_base < N; n_base += BLOCK_K_MED) {

        // Carica dY_int8 vettorizzato (16 bytes = 16 int8 per thread)
        #pragma unroll
        for (int i = tid; i < (BLOCK_M_MED * BLOCK_K_MED) / 16; i += THREADS_MED) {
            int m_l = (i * 16) / BLOCK_K_MED;
            int n_l = (i * 16) % BLOCK_K_MED;
            int m_g = block_m + m_l;
            int n_g = n_base + n_l;

            int4 val = {0, 0, 0, 0};
            // int4 (16B) richiede N%16==0 (n_g è multiplo di 16); altrimenti scalare.
            if ((N % 16 == 0) && m_g < M && n_g + 15 < N) {
                val = *reinterpret_cast<const int4*>(&dY_int8[m_g * N + n_g]);
            } else {
                __align__(16) int8_t tmp[16] = {0};
                for(int j=0; j<16; j++) {
                    if (m_g < M && n_g + j < N) tmp[j] = dY_int8[m_g * N + n_g + j];
                }
                val = *reinterpret_cast<int4*>(&tmp[0]);
            }
            *reinterpret_cast<int4*>(&dY_smem[m_l][n_l]) = val;
        }

        // Espandi W_packed vettorizzato (4 bytes = 16 pesi per thread)
        #pragma unroll
        for (int i = tid; i < (BLOCK_K_MED * BLOCK_N_MED) / 16; i += THREADS_MED) {
            int n_l    = i / (BLOCK_N_MED / 16);
            int byte_chunk_l = i % (BLOCK_N_MED / 16);
            int n_g    = n_base + n_l;
            int byte_g_start = (block_k / 4) + byte_chunk_l * 4;

            uint32_t packed_val = 0;
            // uint32 (4B) richiede packed_K_dim%4==0 (byte_g_start è multiplo di 4).
            if ((packed_K_dim % 4 == 0) && n_g < N && byte_g_start * 4 + 15 < K) {
                packed_val = *reinterpret_cast<const uint32_t*>(&W_packed[n_g * packed_K_dim + byte_g_start]);
            } else {
                __align__(4) uint8_t tmp[4] = {0};
                for(int j=0; j<4; j++) {
                    if (n_g < N && (byte_g_start + j)*4 < K) 
                        tmp[j] = W_packed[n_g * packed_K_dim + byte_g_start + j];
                }
                packed_val = *reinterpret_cast<uint32_t*>(&tmp[0]);
            }

            uint8_t* bytes = (uint8_t*)&packed_val;
            
            #pragma unroll
            for(int b=0; b<4; b++) {
                __align__(4) int8_t exp[4];  // letto via reinterpret_cast<int*>
                expand_int2_x4_int8_bw(bytes[b], exp);
                int k_bl = (byte_chunk_l * 4 + b) * 4;
                if (k_bl + 3 < BLOCK_N_MED) {
                    *reinterpret_cast<int*>(&W_smem[n_l][k_bl]) = *reinterpret_cast<int*>(&exp[0]);
                } else {
                    #pragma unroll
                    for (int j=0; j<4; j++) {
                        if (k_bl + j < BLOCK_N_MED) W_smem[n_l][k_bl + j] = exp[j];
                    }
                }
            }
        }

        __syncthreads();

        // WMMA: ogni warp calcola 2×2 tile = 32×32 output (BLOCK dimezzato vs v2, stessa struttura)
        #pragma unroll
        for (int nn = 0; nn < BLOCK_K_MED; nn += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> a_frag[2];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, wmma::row_major> b_frag[2];

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a_frag[mi],
                    &dY_smem[warp_m * 32 + mi * WMMA_M][nn],
                    BLOCK_K_MED + SMEM_PAD_DY);

            #pragma unroll
            for (int ki = 0; ki < 2; ki++)
                wmma::load_matrix_sync(b_frag[ki],
                    &W_smem[nn][warp_n * 32 + ki * WMMA_N],
                    BLOCK_N_MED + SMEM_PAD_W);

            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ki = 0; ki < 2; ki++)
                    wmma::mma_sync(acc[mi][ki], a_frag[mi], b_frag[ki], acc[mi][ki]);
        }

        __syncthreads();
    }

    // Store: INT32 * scale → FP16 → global
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ki = 0; ki < 2; ki++) {
            int out_m = block_m + warp_m * 32 + mi * WMMA_M;
            int out_k = block_k + warp_n * 32 + ki * WMMA_N;

            wmma::store_matrix_sync(
                &out_tiles[warp_id][0][0], acc[mi][ki],
                WMMA_N + SMEM_PAD_OUT, wmma::mem_row_major);
            __syncwarp();

            int row = lane_id / 2;
            int col_chunk = lane_id % 2;
            int m_o = out_m + row;
            int k_o = out_k + col_chunk * 8;

            // float4 (16B) su dX richiede K%8==0 (k_o è già multiplo di 8).
            if ((K % 8 == 0) && m_o < M && k_o + 7 < K) {
                half out_vec[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    out_vec[i] = __float2half((float)out_tiles[warp_id][row][col_chunk * 8 + i] * out_scale);
                }
                *reinterpret_cast<float4*>(&dX[m_o * K + k_o]) = *reinterpret_cast<float4*>(&out_vec[0]);
            } else {
                for (int i = 0; i < 8; i++) {
                    if (m_o < M && k_o + i < K) {
                        dX[m_o * K + k_o + i] = __float2half((float)out_tiles[warp_id][row][col_chunk * 8 + i] * out_scale);
                    }
                }
            }
        }
    }
}

__global__ void int2_backward_dx_int8_tc_v2_simple_kernel(
    const int8_t*  __restrict__ dY_int8,
    const uint8_t* __restrict__ W_packed,
    half*          __restrict__ dX,
    const float*   __restrict__ d_scale_dy,
    const float gamma,
    const int M, const int N, const int K
) {
    const float out_scale  = (*d_scale_dy / 127.0f) * gamma;
    const int packed_K_dim = (K + 3) / 4;

    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M || k >= K) return;

    int32_t sum = 0;
    for (int n = 0; n < N; n++) {
        int8_t w = bitpack::decode_int2_bits((W_packed[n * packed_K_dim + k/4] >> ((k%4)*2)) & 0x3);
        sum += (int32_t)dY_int8[m * N + n] * (int32_t)w;
    }
    dX[m * K + k] = __float2half((float)sum * out_scale);
}

// ============================================================
// Device properties cache (lazy init)
// ============================================================

static int s_bwd_device = -1;
static int s_bwd_num_sms = -1;

static inline int _bwd_get_num_sms() {
    int device = 0;
    CUDA_CHECK_RETURN(cudaGetDevice(&device), 1);
    if (s_bwd_num_sms >= 0 && s_bwd_device == device) return s_bwd_num_sms;

    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device), 1);
    s_bwd_device = device;
    s_bwd_num_sms = prop.multiProcessorCount;
    return s_bwd_num_sms;
}

extern "C" {

void int2_backward_dx_int8_tc_v2(
    const int8_t*  dY_int8,
    const uint8_t* W_packed,
    half*          dX,
    const float*   d_scale_dy,
    float          gamma,
    int M, int N, int K,
    cudaStream_t   stream
) {
    int num_sms = _bwd_get_num_sms();

    // Output è [M, K] → grid su (K, M)
    int blocks_large = ((K + BLOCK_N - 1) / BLOCK_N) * ((M + BLOCK_M - 1) / BLOCK_M);
    int blocks_med   = ((K + BLOCK_N_MED - 1) / BLOCK_N_MED) * ((M + BLOCK_M_MED - 1) / BLOCK_M_MED);

    if (blocks_large >= num_sms * 2 && M >= BLOCK_M && K >= BLOCK_N && N >= BLOCK_K) {
        dim3 grid((K + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
        int2_backward_dx_int8_tc_v2_kernel<<<grid, THREADS_PER_BLOCK, 0, stream>>>(
            dY_int8, W_packed, dX, d_scale_dy, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());

    } else if (blocks_med >= num_sms && M >= BLOCK_M_MED && K >= BLOCK_N_MED && N >= BLOCK_K_MED) {
        dim3 grid((K + BLOCK_N_MED - 1) / BLOCK_N_MED, (M + BLOCK_M_MED - 1) / BLOCK_M_MED);
        int2_backward_dx_int8_tc_v2_medium_kernel<<<grid, THREADS_MED, 0, stream>>>(
            dY_int8, W_packed, dX, d_scale_dy, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());

    } else {
        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (M + 15) / 16);
        int2_backward_dx_int8_tc_v2_simple_kernel<<<grid, block, 0, stream>>>(
            dY_int8, W_packed, dX, d_scale_dy, gamma, M, N, K);
        CUDA_CHECK(cudaGetLastError());
    }
}

}  // extern "C"
