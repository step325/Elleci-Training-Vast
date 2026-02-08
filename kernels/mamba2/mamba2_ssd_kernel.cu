
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Constants
#define CHUNK_SIZE 64
#define WARP_SIZE 32
#define MAX_D_STATE 128

// -------------------------------------------------------------------------
// Helper for Half Precision Atomics
// -------------------------------------------------------------------------

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700
// Fallback for older cards (CAS loop) not needed for RTX 4070 (sm_89)
// But to be safe and avoid compilation errors on linking:
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int* address_as_ui = (unsigned int*)((char*)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short h = (size_t)address & 2 ? (old >> 16) : (old & 0xFFFF);
        half r = __float2half(__half2float(__ushort_as_half(h)) + __half2float((__half)val));
        unsigned int new_val = (size_t)address & 2 ? (old & 0xFFFF) | ((unsigned int)__half_as_ushort(r) << 16) : (old & 0xFFFF0000) | (unsigned int)__half_as_ushort(r);
        old = atomicCAS(address_as_ui, assumed, new_val);
    } while (assumed != old);
}
#else
// For Pascal+ (sm_60+) half atomicAdd is supported via __half
// c10::Half is layout compatible with __half
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), (__half)val);
}
#endif

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
    int r = threadIdx.x; 

    if (r >= CHUNK_SIZE) return;

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

    // 1. Shared dt cumsum
    __shared__ float dt_cumsum_shared[CHUNK_SIZE];
    float dt_val = (float)dt[dt_offset + r * stride_dt_cs + h];
    dt_cumsum_shared[r] = dt_val;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<CHUNK_SIZE; ++i) {
            sum += dt_cumsum_shared[i];
            dt_cumsum_shared[i] = sum;
        }
    }
    __syncthreads();
    
    extern __shared__ float shared_mem[];
    float* B_shared = shared_mem;
    float* K_shared = &B_shared[CHUNK_SIZE * d_state]; 

    // Load B
    for (int p = 0; p < d_state; ++p) {
        B_shared[r * d_state + p] = (float)B_param[B_offset + r * stride_B_cs + p];
    }
    __syncthreads();

    // Compute K = C . B^T
    for (int c = 0; c < CHUNK_SIZE; ++c) {
        float dot = 0.0f;
        for (int p = 0; p < d_state; ++p) {
            float b_val = B_shared[c * d_state + p];
            float c_val = (float)C_param[C_base + r * d_state + p]; 
            dot += b_val * c_val;
        }
        K_shared[r * CHUNK_SIZE + c] = dot;
    }
    __syncthreads();
    
    // Forward
    for (int d = 0; d < head_dim; ++d) {
        float y_acc = 0.0f;
        for (int c = 0; c <= r; ++c) {
            float dt_r = dt_cumsum_shared[r];
            float dt_c = dt_cumsum_shared[c];
            float m_val_log = a_val * (dt_r - dt_c);
            if (m_val_log > 0.0f) m_val_log = 0.0f; // Clamp
            float m_val = expf(m_val_log);
            
            float dt_c_raw = (c==0) ? dt_cumsum_shared[0] : dt_cumsum_shared[c] - dt_cumsum_shared[c-1];
            
            long x_idx_c = x_offset + c * stride_x_cs + h * stride_x_h + d;
            float x_val = (float)X[x_idx_c];
            
            float k_val = K_shared[r * CHUNK_SIZE + c];
            y_acc += m_val * (x_val * dt_c_raw) * k_val;
        }
        long y_idx = x_offset + r * stride_x_cs + h * stride_x_h + d; 
        Y[y_idx] = (scalar_t)y_acc;
    }
}

template <typename scalar_t>
__global__ void mamba2_ssd_bwd_kernel(
    const scalar_t* __restrict__ dY,
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ dt,
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B_param,
    const scalar_t* __restrict__ C_param,
    scalar_t* __restrict__ dX,
    scalar_t* __restrict__ ddt,
    scalar_t* __restrict__ dA,
    scalar_t* __restrict__ dB,
    scalar_t* __restrict__ dC,
    const int batch, const int n_chunks, const int n_heads, const int head_dim, const int d_state
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int c_idx = blockIdx.z;
    int r = threadIdx.x; 

    if (r >= CHUNK_SIZE) return;

    // Strides
    int stride_x_b  = n_chunks * CHUNK_SIZE * n_heads * head_dim;
    int stride_x_c  = CHUNK_SIZE * n_heads * head_dim;
    int stride_x_cs = n_heads * head_dim;
    int stride_x_h  = head_dim;
    
    int stride_dt_cs = n_heads;
    int stride_B_cs = d_state;

    long x_base  = b * stride_x_b + c_idx * stride_x_c;
    long dt_base = b * (n_chunks * CHUNK_SIZE * n_heads) + c_idx * (CHUNK_SIZE * n_heads);
    long B_base  = b * (n_chunks * CHUNK_SIZE * d_state) + c_idx * (CHUNK_SIZE * d_state);
    
    float a_val = -expf((float)A[h]); // A_log input

    __shared__ float dt_cumsum_shared[CHUNK_SIZE];
    float dt_val = (float)dt[dt_base + r * stride_dt_cs + h];
    dt_cumsum_shared[r] = dt_val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for(int i=0; i<CHUNK_SIZE; ++i) {
            sum += dt_cumsum_shared[i];
            dt_cumsum_shared[i] = sum;
        }
    }
    __syncthreads();

    extern __shared__ float shared_mem[];
    float* B_shared = shared_mem;
    float* C_shared = &B_shared[CHUNK_SIZE * d_state];

    // Load B and C
    for (int p = 0; p < d_state; ++p) {
        B_shared[r * d_state + p] = (float)B_param[B_base + r * stride_B_cs + p];
        C_shared[r * d_state + p] = (float)C_param[B_base + r * stride_B_cs + p]; 
    }
    __syncthreads();
    
    for (int d = 0; d < head_dim; ++d) {
        float dx_acc = 0.0f;
        float ddt_acc = 0.0f;
        
        for (int k = r; k < CHUNK_SIZE; ++k) {
            float spread = dt_cumsum_shared[k] - dt_cumsum_shared[r];
            float m_val_log = a_val * spread;
            if (m_val_log > 0.0f) m_val_log = 0.0f;
            float m_val = expf(m_val_log);
            
            float k_val = 0.0f;
            for(int p=0; p<d_state; ++p) {
                k_val += C_shared[k*d_state+p] * B_shared[r*d_state+p];
            }
            
            long dy_idx = x_base + k * stride_x_cs + h * stride_x_h + d;
            float dy_val = (float)dY[dy_idx];
            
            dx_acc += dy_val * m_val * k_val;
            
            long x_idx = x_base + r * stride_x_cs + h * stride_x_h + d;
            float x_val = (float)X[x_idx];
            ddt_acc += dy_val * m_val * x_val * k_val; 
        }
        
        dx_acc *= dt_val; 
        long dx_out_idx = x_base + r * stride_x_cs + h * stride_x_h + d;
        dX[dx_out_idx] = (scalar_t)dx_acc;
        
        long ddt_idx = dt_base + r * stride_dt_cs + h;
        atomicAdd(&ddt[ddt_idx], (scalar_t)ddt_acc);
    }
}

// -------------------------------------------------------------------------
// LAUNCHERS (Compiled by NVCC)
// -------------------------------------------------------------------------

torch::Tensor mamba2_ssd_fwd_cuda_launcher(
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    int batch = X.size(0);
    int n_chunks = X.size(1);
    int chunk_size = X.size(2);
    int n_heads = X.size(3);
    int head_dim = X.size(4);
    int d_state = B_param.size(3);

    auto Y = torch::zeros_like(X);

    dim3 threads(chunk_size);
    dim3 blocks(batch, n_heads, n_chunks);
    
    // Shared mem: B (CS*P) + K (CS*CS) or B+C (2*CS*P)?
    // Forward uses B+K. P=128 => B=32KB, K=16KB -> 48KB.
    int shared_mem_size = (chunk_size * d_state + chunk_size * chunk_size) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "mamba2_ssd_fwd", ([&] {
        mamba2_ssd_fwd_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            X.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B_param.data_ptr<scalar_t>(),
            C_param.data_ptr<scalar_t>(),
            Y.data_ptr<scalar_t>(),
            batch, n_chunks, n_heads, head_dim, d_state
        );
    }));
    
    return Y;
}

std::vector<torch::Tensor> mamba2_ssd_bwd_cuda_launcher(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    int batch = X.size(0);
    int n_chunks = X.size(1);
    int chunk_size = X.size(2);
    int n_heads = X.size(3);
    int head_dim = X.size(4);
    int d_state = B_param.size(3);

    auto dX = torch::zeros_like(X);
    auto ddt = torch::zeros_like(dt);
    auto dA = torch::zeros_like(A);
    auto dB = torch::zeros_like(B_param);
    auto dC = torch::zeros_like(C_param);

    dim3 threads(chunk_size);
    dim3 blocks(batch, n_heads, n_chunks);
    
    // Backward uses B+C. 2 * 64 * 128 * 4 = 64KB.
    int shared_mem_size = (2 * chunk_size * d_state) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(X.scalar_type(), "mamba2_ssd_bwd", ([&] {
        mamba2_ssd_bwd_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            dY.data_ptr<scalar_t>(),
            X.data_ptr<scalar_t>(),
            dt.data_ptr<scalar_t>(),
            A.data_ptr<scalar_t>(),
            B_param.data_ptr<scalar_t>(),
            C_param.data_ptr<scalar_t>(),
            dX.data_ptr<scalar_t>(),
            ddt.data_ptr<scalar_t>(),
            dA.data_ptr<scalar_t>(),
            dB.data_ptr<scalar_t>(),
            dC.data_ptr<scalar_t>(),
            batch, n_chunks, n_heads, head_dim, d_state
        );
    }));

    return {dX, ddt, dA, dB, dC};
}
