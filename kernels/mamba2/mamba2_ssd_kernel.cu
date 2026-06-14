
// NB: <torch/extension.h> NON va incluso in questo .cu — verrebbe parsato da
// nvcc e, con GCC >= 15/16, rompe su ATen/core/List_inl.h (function_schema.h).
// La parte torch (Tensor / dispatch / stream) vive in mamba2_ssd.cpp; qui
// esponiamo solo launcher a puntatori grezzi. <c10/util/Half.h> è leggero e
// fornisce il tipo c10::Half senza trascinare la dispatch machinery.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>

// Constants
#define CHUNK_SIZE 64
#define WARP_SIZE 32
#define MAX_D_STATE 128
#define D_THREADS 16   // Parallelismo su head_dim: ogni thread gestisce head_dim/D_THREADS slices

// NB: l'helper atomicAdd(c10::Half) è stato rimosso — il backward è ora interamente
// tensor-ops in mamba2_ssd.cpp e il forward non usa atomici. Nessun consumatore.

// -------------------------------------------------------------------------
// KERNELS
// -------------------------------------------------------------------------

template <typename scalar_t>
__global__ void mamba2_ssd_fwd_kernel(
    const scalar_t* __restrict__ X,          // [B, NC, CS, H, D]
    const scalar_t* __restrict__ dt,         // [B, NC, CS, H]
    const scalar_t* __restrict__ A,          // [H]
    const scalar_t* __restrict__ B_param,    // [B, NC, CS, P]
    const scalar_t* __restrict__ C_param,    // [B, NC, CS, P]
    scalar_t* __restrict__ Y,                // [B, n_chunks, CS, H, D]
    const int batch,
    const int n_chunks,
    const int n_heads,
    const int head_dim,
    const int d_state
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int c_idx = blockIdx.z;
    int r     = threadIdx.x;  // 0..CHUNK_SIZE-1 (posizione nel chunk)
    int d_tid = threadIdx.y;  // 0..D_THREADS-1  (slice di head_dim)

    // NB: il launcher impone threads.x == CHUNK_SIZE, quindi r < CHUNK_SIZE sempre.
    // Nessun early-return: ogni thread deve raggiungere i __syncthreads sotto (un
    // return condizionale prima di una barriera causerebbe deadlock).

    // Strides
    int stride_x_b  = n_chunks * CHUNK_SIZE * n_heads * head_dim;
    int stride_x_c  = CHUNK_SIZE * n_heads * head_dim;
    int stride_x_cs = n_heads * head_dim;
    int stride_x_h  = head_dim;

    int stride_dt_b  = n_chunks * CHUNK_SIZE * n_heads;
    int stride_dt_c  = CHUNK_SIZE * n_heads;
    int stride_dt_cs = n_heads;

    int stride_B_b = n_chunks * CHUNK_SIZE * d_state;
    int stride_B_c = CHUNK_SIZE * d_state;
    int stride_B_cs = d_state;

    long x_offset  = b * stride_x_b + c_idx * stride_x_c;
    long dt_offset = b * stride_dt_b + c_idx * stride_dt_c;
    long B_offset  = b * stride_B_b + c_idx * stride_B_c;
    long C_base = b * stride_B_b + c_idx * stride_B_c;

    // A input is assumed to be A_log. A_val = -exp(A_log)
    float a_val = -expf((float)A[h]);

    // 1. Shared dt cumsum — solo d_tid==0 carica
    __shared__ float dt_cumsum_shared[CHUNK_SIZE];
    if (d_tid == 0) {
        dt_cumsum_shared[r] = (float)dt[dt_offset + r * stride_dt_cs + h];
    }
    __syncthreads();

    if (threadIdx.x == 0 && d_tid == 0) {
        float sum = 0;
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            sum += dt_cumsum_shared[i];
            dt_cumsum_shared[i] = sum;
        }
    }
    __syncthreads();

    extern __shared__ float shared_mem[];
    float* B_shared = shared_mem;
    float* K_shared = &B_shared[CHUNK_SIZE * d_state];

    // Load B — solo d_tid==0
    if (d_tid == 0) {
        for (int p = 0; p < d_state; ++p) {
            B_shared[r * d_state + p] = (float)B_param[B_offset + r * stride_B_cs + p];
        }
    }
    __syncthreads();

    // Compute K = C . B^T — solo d_tid==0
    if (d_tid == 0) {
        for (int c = 0; c < CHUNK_SIZE; ++c) {
            float dot = 0.0f;
            for (int p = 0; p < d_state; ++p) {
                float b_val = B_shared[c * d_state + p];
                float c_val = (float)C_param[C_base + r * d_state + p];
                dot += b_val * c_val;
            }
            K_shared[r * CHUNK_SIZE + c] = dot;
        }
    }
    __syncthreads();

    // Forward: loop parallelo su d con stride D_THREADS
    for (int d = d_tid; d < head_dim; d += D_THREADS) {
        float y_acc = 0.0f;
        for (int c = 0; c <= r; ++c) {
            float dt_r = dt_cumsum_shared[r];
            float dt_c = dt_cumsum_shared[c];
            float m_val_log = a_val * (dt_r - dt_c);
            if (m_val_log > 0.0f) m_val_log = 0.0f; // Clamp

            float m_val = expf(m_val_log);

            float dt_c_raw = (c == 0) ? dt_cumsum_shared[0]
                                       : dt_cumsum_shared[c] - dt_cumsum_shared[c - 1];

            long x_idx_c = x_offset + c * stride_x_cs + h * stride_x_h + d;
            float x_val = (float)X[x_idx_c];

            float k_val = K_shared[r * CHUNK_SIZE + c];
            y_acc += m_val * (x_val * dt_c_raw) * k_val;
        }
        long y_idx = x_offset + r * stride_x_cs + h * stride_x_h + d;
        Y[y_idx] = (scalar_t)y_acc;
    }
}

// -------------------------------------------------------------------------
// LAUNCHER A PUNTATORI GREZZI (compilati da NVCC, niente tipi torch)
// Il dispatch sul dtype e la gestione dei tensori avviene in mamba2_ssd.cpp.
// dtype: 0 = float32, 1 = float64, 2 = float16 (c10::Half)
// -------------------------------------------------------------------------

void mamba2_ssd_fwd_launch(
    const void* X, const void* dt, const void* A,
    const void* B_param, const void* C_param, void* Y,
    int dtype, int batch, int n_chunks, int n_heads, int head_dim, int d_state,
    cudaStream_t stream
) {
    // CHUNK_SIZE threads su x (posizione), D_THREADS su y (slice head_dim)
    dim3 threads(CHUNK_SIZE, D_THREADS);
    dim3 blocks(batch, n_heads, n_chunks);
    // Shared mem: B (CS*P) + K (CS*CS)
    int shared_mem_size = (CHUNK_SIZE * d_state + CHUNK_SIZE * CHUNK_SIZE) * sizeof(float);

    // M2: l'opt-in serve quando la shared TOTALE supera i 48KB di default. La dinamica
    // (shared_mem_size) va sommata alla statica dt_cumsum_shared[CHUNK_SIZE] (256B),
    // altrimenti a d_state=128 (dinamica = 49152 = 48KB esatti) il lancio fallirebbe.
    // sm_8x arriva a ~100KB con l'opt-in.
    const int static_smem = CHUNK_SIZE * (int)sizeof(float);  // dt_cumsum_shared
    #define MAMBA2_FWD(T)                                                          \
        do {                                                                       \
            if (shared_mem_size + static_smem > 49152)                             \
                cudaFuncSetAttribute((const void*)mamba2_ssd_fwd_kernel<T>,        \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size); \
            mamba2_ssd_fwd_kernel<T><<<blocks, threads, shared_mem_size, stream>>>(\
                (const T*)X, (const T*)dt, (const T*)A, (const T*)B_param,         \
                (const T*)C_param, (T*)Y,                                          \
                batch, n_chunks, n_heads, head_dim, d_state);                      \
        } while (0)
    switch (dtype) {
        case 0: MAMBA2_FWD(float);     break;
        case 1: MAMBA2_FWD(double);    break;
        case 2: MAMBA2_FWD(c10::Half); break;
    }
    #undef MAMBA2_FWD
}
