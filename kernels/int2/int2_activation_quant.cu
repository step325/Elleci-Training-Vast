/**
 * INT8 Activation Quantization/Dequantization Kernels
 *
 * For compressing saved_input in INT2 layers to reduce memory footprint.
 * Uses per-tensor symmetric quantization: x_int8 = round(x / scale)
 * where scale = max(abs(x)) / 127
 *
 * Target: NVIDIA Ada Lovelace (SM 8.9)
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Configuration
// ============================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// ============================================================
// Reduction Utilities
// ============================================================

/**
 * Warp-level max reduction
 */
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

/**
 * Block-level max reduction using shared memory
 */
__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp reduction
    val = warp_reduce_max(val);

    // Write warp result to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;

    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// ============================================================
// Quantization Kernels
// ============================================================

/**
 * Pass 1: Find max absolute value per tensor
 * Each block processes a chunk and atomically updates global max
 */
__global__ void find_absmax_kernel(
    const half* __restrict__ input,
    float* __restrict__ global_max,
    const int size
) {
    __shared__ float shared_max[BLOCK_SIZE / WARP_SIZE];

    float local_max = 0.0f;

    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {
        float val = fabsf(__half2float(input[idx]));
        local_max = fmaxf(local_max, val);
    }

    // Block reduction
    float block_max = block_reduce_max(local_max, shared_max);

    // Atomic update global max (only thread 0)
    if (threadIdx.x == 0 && block_max > 0.0f) {
        // Use atomicMax with float reinterpretation
        // This works because positive floats compare correctly as ints
        int* global_max_int = (int*)global_max;
        int block_max_int = __float_as_int(block_max);
        atomicMax(global_max_int, block_max_int);
    }
}

/**
 * Pass 2: Quantize FP16 to INT8 using precomputed scale
 */
__global__ void quantize_kernel(
    const half* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scale_ptr,
    const int size
) {
    float scale = *scale_ptr;
    float inv_scale = (scale > 1e-8f) ? (127.0f / scale) : 0.0f;

    // Grid-stride loop with vectorization
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time when possible
    for (int i = idx * 4; i < size - 3; i += stride * 4) {
        // Load 4 half values
        float4 vals;
        vals.x = __half2float(input[i]);
        vals.y = __half2float(input[i + 1]);
        vals.z = __half2float(input[i + 2]);
        vals.w = __half2float(input[i + 3]);

        // Quantize
        int8_t q0 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(vals.x * inv_scale)));
        int8_t q1 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(vals.y * inv_scale)));
        int8_t q2 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(vals.z * inv_scale)));
        int8_t q3 = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(vals.w * inv_scale)));

        // Store
        output[i] = q0;
        output[i + 1] = q1;
        output[i + 2] = q2;
        output[i + 3] = q3;
    }

    // Handle remainder
    for (int i = idx + (size / 4) * 4; i < size; i += stride) {
        float val = __half2float(input[i]);
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(val * inv_scale)));
    }
}

/**
 * Dequantize INT8 to FP16 using scale
 */
__global__ void dequantize_kernel(
    const int8_t* __restrict__ input,
    half* __restrict__ output,
    const float* __restrict__ scale_ptr,
    const int size
) {
    float scale_127 = (*scale_ptr) / 127.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time
    for (int i = idx * 4; i < size - 3; i += stride * 4) {
        // Load and dequantize
        float v0 = (float)input[i] * scale_127;
        float v1 = (float)input[i + 1] * scale_127;
        float v2 = (float)input[i + 2] * scale_127;
        float v3 = (float)input[i + 3] * scale_127;

        // Store as half
        output[i] = __float2half(v0);
        output[i + 1] = __float2half(v1);
        output[i + 2] = __float2half(v2);
        output[i + 3] = __float2half(v3);
    }

    // Handle remainder
    for (int i = idx + (size / 4) * 4; i < size; i += stride) {
        output[i] = __float2half((float)input[i] * scale_127);
    }
}

/**
 * Fused quantize: finds absmax and quantizes in optimized passes
 */
__global__ void quantize_fused_pass2_kernel(
    const half* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale_out,
    const float absmax,
    const int size
) {
    // Compute scale once
    float scale = absmax;
    float inv_scale = (scale > 1e-8f) ? (127.0f / scale) : 0.0f;

    // Store scale (only first thread of first block)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scale_out = scale;
    }

    // Quantize
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        float val = __half2float(input[i]);
        output[i] = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(val * inv_scale)));
    }
}

// ============================================================
// C++ Interface Functions
// ============================================================

extern "C" {

/**
 * Quantize FP16 tensor to INT8
 *
 * @param input     Input FP16 tensor [size]
 * @param output    Output INT8 tensor [size]
 * @param scale_out Output scale (scalar on device)
 * @param size      Total number of elements
 * @param stream    CUDA stream
 */
void quantize_activation(
    const half* input,
    int8_t* output,
    float* scale_out,
    const int size,
    cudaStream_t stream
) {
    // Use device memory for global max
    float* d_global_max;
    cudaMallocAsync(&d_global_max, sizeof(float), stream);
    cudaMemsetAsync(d_global_max, 0, sizeof(float), stream);

    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 1024);  // Cap at 1024 blocks

    // Pass 1: Find absmax
    find_absmax_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, d_global_max, size
    );

    // Copy absmax back (we need it on host briefly for launch)
    int h_absmax_int;
    cudaMemcpyAsync(&h_absmax_int, d_global_max, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Reinterpret the atomicMax result (using C++ union for host code)
    union { int i; float f; } absmax_union;
    absmax_union.i = h_absmax_int;
    float h_absmax = absmax_union.f;

    // Pass 2: Quantize with computed scale
    quantize_fused_pass2_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, scale_out, h_absmax, size
    );

    cudaFreeAsync(d_global_max, stream);
}

/**
 * Dequantize INT8 tensor to FP16
 *
 * @param input     Input INT8 tensor [size]
 * @param output    Output FP16 tensor [size]
 * @param scale     Scale factor (host value)
 * @param size      Total number of elements
 * @param stream    CUDA stream
 */
void dequantize_activation(
    const int8_t* input,
    half* output,
    const float* scale_ptr,
    const int size,
    cudaStream_t stream
) {
    int num_blocks = (size + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    num_blocks = min(num_blocks, 1024);

    dequantize_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, scale_ptr, size
    );
}

/**
 * Async version: Quantize without synchronization
 * Returns immediately, scale is written to scale_out on device
 */
void quantize_activation_async(
    const half* input,
    int8_t* output,
    float* scale_out,
    float* d_global_max,  // Pre-allocated device memory for absmax
    const int size,
    cudaStream_t stream
) {
    // Reset global max
    cudaMemsetAsync(d_global_max, 0, sizeof(float), stream);

    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    num_blocks = min(num_blocks, 1024);

    // Pass 1: Find absmax
    find_absmax_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, d_global_max, size
    );

    // Pass 2: Quantize (reads absmax from device memory)
    quantize_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input, output, d_global_max, size
    );

    // Copy scale to output
    cudaMemcpyAsync(scale_out, d_global_max, sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

}  // extern "C"
