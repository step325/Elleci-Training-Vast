
#include <torch/extension.h>
#include <vector>

// Declarations of functions implemented in .cu
torch::Tensor mamba2_ssd_fwd_cuda_launcher(
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
);

std::vector<torch::Tensor> mamba2_ssd_bwd_cuda_launcher(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
);

// Python Wrappers
torch::Tensor mamba2_ssd_fwd(
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    return mamba2_ssd_fwd_cuda_launcher(X, dt, A, B_param, C_param);
}

std::vector<torch::Tensor> mamba2_ssd_bwd(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    return mamba2_ssd_bwd_cuda_launcher(dY, X, dt, A, B_param, C_param);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mamba2_ssd_fwd, "Mamba-2 SSD Forward (Fused)");
  m.def("backward", &mamba2_ssd_bwd, "Mamba-2 SSD Backward (Fused Recompute)");
}
