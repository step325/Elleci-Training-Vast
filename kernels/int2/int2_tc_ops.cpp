/**
 * PyTorch C++ Extension for INT2 Tensor Core Kernels
 *
 * Provides Python bindings for the optimized TC kernels:
 * - int2_matmul_tc: Forward pass with Tensor Cores
 * - int2_backward_dx_tc: Backward pass with Tensor Cores
 * - int2_hysteresis_optimized_update: Optimized hysteresis update
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Forward declarations of CUDA kernels
extern "C" {
    void int2_matmul_tc(
        const half* X, const uint8_t* W_packed, half* Y,
        float gamma, int M, int N, int K, cudaStream_t stream
    );

    void int2_backward_dx_tc(
        const half* dY, const uint8_t* W_packed, half* dX,
        float gamma, int M, int N, int K, cudaStream_t stream
    );

    void int2_hysteresis_optimized_update(
        const half* dY, const half* X, int M,
        uint8_t* W_packed, uint8_t* H_packed,
        float lr, float lr_scale, int threshold, float decay_rate,
        int step, int N, int K, cudaStream_t stream
    );

    // ========================================================================
    // HYSTERESIS V2: Uses pre-computed gradient (from cuBLAS)
    // This is ~9x faster than the fused version because:
    // - dW is computed via torch.mm (uses Tensor Cores)
    // - This kernel is O(N×K) instead of O(M×N×K)
    // ========================================================================
    void int2_hysteresis_v2_update(
        const float* dW,       // Pre-computed gradient [N, K] from cuBLAS
        uint8_t* W_packed,
        uint8_t* H_packed,
        float lr, float lr_scale, int threshold, float decay_rate,
        int step, int N, int K, cudaStream_t stream
    );

    // INT2 Unpack kernel: W_packed → W_fp16
    void int2_unpack_to_fp16(
        const uint8_t* W_packed, half* W_fp16,
        int N, int K, cudaStream_t stream
    );

    // Activation quantization kernels
    void quantize_activation(
        const half* input, int8_t* output, float* scale_out,
        int size, cudaStream_t stream
    );

    void dequantize_activation(
        const int8_t* input, half* output, const float* scale_ptr,
        int size, cudaStream_t stream
    );

    void quantize_activation_async(
        const half* input, int8_t* output, float* scale_out,
        float* d_global_max, int size, cudaStream_t stream
    );
}


/**
 * Forward pass with Tensor Cores: Y = X @ W^T * gamma
 */
torch::Tensor tc_matmul(
    torch::Tensor X,        // [*, K] float16
    torch::Tensor W_packed, // [N, K/4] uint8
    double gamma
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat16, "X must be float16");
    TORCH_CHECK(W_packed.dtype() == torch::kUInt8, "W_packed must be uint8");

    int N = W_packed.size(0);
    int packed_K = W_packed.size(1);
    int K = packed_K * 4;

    auto X_flat = X.view({-1, X.size(-1)});
    int M = X_flat.size(0);

    TORCH_CHECK(X_flat.size(1) == K,
        "Dimension mismatch: X has ", X_flat.size(1), " features but W expects ", K);

    auto Y = torch::empty({M, N}, X.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int2_matmul_tc(
        (const half*)X_flat.data_ptr(),
        (const uint8_t*)W_packed.data_ptr(),
        (half*)Y.data_ptr(),
        (float)gamma,
        M, N, K,
        stream
    );

    auto out_shape = X.sizes().vec();
    out_shape.back() = N;
    return Y.view(out_shape);
}

/**
 * Backward pass with Tensor Cores: dX = dY @ W * gamma
 */
torch::Tensor tc_backward_input(
    torch::Tensor dY,       // [*, N] float16
    torch::Tensor W_packed, // [N, K/4] uint8
    double gamma,
    int64_t K
) {
    TORCH_CHECK(dY.is_cuda(), "dY must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");

    int N = W_packed.size(0);
    auto dY_flat = dY.view({-1, dY.size(-1)});
    int M = dY_flat.size(0);

    auto dX = torch::empty({M, K}, dY.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int2_backward_dx_tc(
        (const half*)dY_flat.data_ptr(),
        (const uint8_t*)W_packed.data_ptr(),
        (half*)dX.data_ptr(),
        (float)gamma,
        M, N, (int)K,
        stream
    );

    auto out_shape = dY.sizes().vec();
    out_shape.back() = K;
    return dX.view(out_shape);
}

/**
 * Hysteresis update step (optimized)
 */
void tc_hysteresis_step(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor W_packed,
    torch::Tensor H_packed,
    double lr,
    double lr_scale,
    int64_t threshold,
    double decay,
    int64_t step
) {
    TORCH_CHECK(dY.is_cuda() && X.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(W_packed.is_cuda() && H_packed.is_cuda(), "State must be on CUDA");

    auto dY_flat = dY.view({-1, dY.size(-1)});
    auto X_flat = X.view({-1, X.size(-1)});

    int M = dY_flat.size(0);
    int N = dY_flat.size(1);
    int K = X_flat.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int2_hysteresis_optimized_update(
        (const half*)dY_flat.data_ptr(),
        (const half*)X_flat.data_ptr(),
        M,
        (uint8_t*)W_packed.data_ptr(),
        (uint8_t*)H_packed.data_ptr(),
        (float)lr,
        (float)lr_scale,
        (int)threshold,
        (float)decay,
        (int)step,
        N, K,
        stream
    );
}

/**
 * Hysteresis update step v2 (OPTIMIZED - uses pre-computed gradient)
 * 
 * ============================================================================
 * This is ~9x faster than tc_hysteresis_step because:
 * - dW is computed externally via torch.mm (cuBLAS Tensor Cores)
 * - This kernel is O(N×K) instead of O(M×N×K)
 * ============================================================================
 */
void tc_hysteresis_step_v2(
    torch::Tensor dW,        // Pre-computed gradient [N, K] from cuBLAS
    torch::Tensor W_packed,
    torch::Tensor H_packed,
    double lr,
    double lr_scale,
    int64_t threshold,
    double decay,
    int64_t step
) {
    TORCH_CHECK(dW.is_cuda(), "dW must be a CUDA tensor");
    TORCH_CHECK(W_packed.is_cuda() && H_packed.is_cuda(), "State must be on CUDA");
    TORCH_CHECK(dW.dtype() == torch::kFloat32, "dW must be float32");

    dW = dW.contiguous();
    int N = dW.size(0);
    int K = dW.size(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int2_hysteresis_v2_update(
        (const float*)dW.data_ptr(),
        (uint8_t*)W_packed.data_ptr(),
        (uint8_t*)H_packed.data_ptr(),
        (float)lr,
        (float)lr_scale,
        (int)threshold,
        (float)decay,
        (int)step,
        N, K,
        stream
    );
}



/**
 * Get memory statistics for the TC kernels
 */
std::string tc_memory_stats() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    char buf[256];
    snprintf(buf, sizeof(buf),
        "GPU Memory: %.2f GB free / %.2f GB total",
        free_mem / 1e9, total_mem / 1e9);
    return std::string(buf);
}

/**
 * Quantize FP16 activations to INT8 (per-tensor symmetric)
 * Returns tuple: (quantized_int8, scale)
 */
std::tuple<torch::Tensor, torch::Tensor> tc_quantize_activation(
    torch::Tensor input
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be float16");

    input = input.contiguous();
    int size = input.numel();

    // Allocate output tensors
    auto output = torch::empty(input.sizes(), torch::dtype(torch::kInt8).device(input.device()));
    auto scale = torch::empty({1}, torch::dtype(torch::kFloat32).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    quantize_activation(
        (const half*)input.data_ptr(),
        (int8_t*)output.data_ptr(),
        (float*)scale.data_ptr(),
        size,
        stream
    );

    return std::make_tuple(output, scale);
}

/**
 * Dequantize INT8 activations back to FP16
 */
torch::Tensor tc_dequantize_activation(
    torch::Tensor input,
    torch::Tensor scale
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Scale must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt8, "Input must be int8");

    input = input.contiguous();
    scale = scale.contiguous();
    int size = input.numel();

    // Allocate output
    auto output = torch::empty(input.sizes(), torch::dtype(torch::kFloat16).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Pass device pointer directly - zero GPU→CPU sync!
    dequantize_activation(
        (const int8_t*)input.data_ptr(),
        (half*)output.data_ptr(),
        (const float*)scale.data_ptr<float>(),
        size,
        stream
    );

    return output;
}

/**
 * Async quantize: no GPU→CPU sync, uses pre-allocated absmax buffer
 * Returns tuple: (quantized_int8, scale)
 */
std::tuple<torch::Tensor, torch::Tensor> tc_quantize_activation_async(
    torch::Tensor input,
    torch::Tensor d_global_max  // Pre-allocated float32 buffer on GPU (size 1)
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat16, "Input must be float16");
    TORCH_CHECK(d_global_max.is_cuda(), "d_global_max must be a CUDA tensor");

    input = input.contiguous();
    int size = input.numel();

    // Allocate output tensors
    auto output = torch::empty(input.sizes(), torch::dtype(torch::kInt8).device(input.device()));
    auto scale = torch::empty({1}, torch::dtype(torch::kFloat32).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Async: no cudaStreamSynchronize, absmax stays on GPU
    quantize_activation_async(
        (const half*)input.data_ptr(),
        (int8_t*)output.data_ptr(),
        (float*)scale.data_ptr(),
        (float*)d_global_max.data_ptr<float>(),
        size,
        stream
    );

    return std::make_tuple(output, scale);
}

/**
 * Unpack INT2 packed weights to FP16 buffer
 * W_packed [N, K/4] uint8 → W_fp16 [N, K] float16
 */
void tc_unpack_int2(
    torch::Tensor W_packed,  // [N, K/4] uint8
    torch::Tensor W_fp16,    // [N, K] float16 (pre-allocated output buffer)
    int64_t K                // unpacked K dimension
) {
    TORCH_CHECK(W_packed.is_cuda(), "W_packed must be a CUDA tensor");
    TORCH_CHECK(W_fp16.is_cuda(), "W_fp16 must be a CUDA tensor");
    TORCH_CHECK(W_packed.dtype() == torch::kUInt8, "W_packed must be uint8");
    TORCH_CHECK(W_fp16.dtype() == torch::kFloat16, "W_fp16 must be float16");

    int N = W_packed.size(0);

    TORCH_CHECK(W_fp16.size(0) == N, "W_fp16 row count must match W_packed");
    TORCH_CHECK(W_fp16.size(1) == K, "W_fp16 col count must equal K");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    int2_unpack_to_fp16(
        (const uint8_t*)W_packed.data_ptr(),
        (half*)W_fp16.data_ptr(),
        N, (int)K,
        stream
    );
}

// PyBind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "INT2 Tensor Core optimized operations";

    m.def("matmul", &tc_matmul,
        "INT2 matrix multiplication with Tensor Cores (Y = X @ W^T * gamma)");

    m.def("backward_input", &tc_backward_input,
        "INT2 backward pass for input with Tensor Cores (dX = dY @ W * gamma)");

    m.def("hysteresis_step", &tc_hysteresis_step,
        "Optimized hysteresis update step (legacy - O(M×N×K))");

    m.def("hysteresis_step_v2", &tc_hysteresis_step_v2,
        "Hysteresis update v2 with pre-computed gradient (fast - O(N×K))");

    m.def("memory_stats", &tc_memory_stats,
        "Get GPU memory statistics");

    m.def("quantize_activation", &tc_quantize_activation,
        "Quantize FP16 activations to INT8 (returns tuple of int8 tensor and scale)");

    m.def("quantize_activation_async", &tc_quantize_activation_async,
        "Async quantize FP16 to INT8 using pre-allocated absmax buffer (zero sync)");

    m.def("dequantize_activation", &tc_dequantize_activation,
        "Dequantize INT8 activations back to FP16");

    m.def("unpack_int2", &tc_unpack_int2,
        "Unpack INT2 weights to FP16 buffer (W_packed [N,K/4] → W_fp16 [N,K])");
}

