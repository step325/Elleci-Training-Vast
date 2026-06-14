
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

// Launcher a puntatori grezzi implementati in mamba2_ssd_kernel.cu.
// dtype: 0 = float32, 1 = float64, 2 = float16 (c10::Half).
// (il backward è interamente tensor-ops in mamba2_ssd_bwd qui sotto: nessun kernel bwd)
void mamba2_ssd_fwd_launch(
    const void* X, const void* dt, const void* A,
    const void* B_param, const void* C_param, void* Y,
    int dtype, int batch, int n_chunks, int n_heads, int head_dim, int d_state,
    cudaStream_t stream);

// Mappa lo scalar_type del tensore sul codice dtype atteso dal .cu.
// Mantiene la parità con AT_DISPATCH_FLOATING_TYPES_AND_HALF (float/double/half).
static int dtype_code(const torch::Tensor& t) {
    auto st = t.scalar_type();
    if (st == torch::kFloat32) return 0;
    if (st == torch::kFloat64) return 1;
    if (st == torch::kFloat16) return 2;
    TORCH_CHECK(false, "mamba2_ssd: dtype non supportato (atteso float32/float64/float16): ", st);
}

// X: [B, NC, CS, H, D] ; B_param/C_param: [B, NC, CS, P]
torch::Tensor mamba2_ssd_fwd(
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    int batch    = X.size(0);
    int n_chunks = X.size(1);
    int n_heads  = X.size(3);
    int head_dim = X.size(4);
    int d_state  = B_param.size(3);

    auto Y = torch::zeros_like(X);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    mamba2_ssd_fwd_launch(
        X.data_ptr(), dt.data_ptr(), A.data_ptr(),
        B_param.data_ptr(), C_param.data_ptr(), Y.data_ptr(),
        dtype_code(X), batch, n_chunks, n_heads, head_dim, d_state, stream);

    return Y;
}

// Backward COMPLETO (M1). A è A_log. Calcola dx, ddt, dA, dB, dC corretti
// ricomponendo gli intermedi (cs, M, K, coef) da (X, dt, A_log, B, C) e usando
// le identità derivate (tutte piccole matmul + cumulate, GPU via cuBLAS):
//   coef[r,c,h] = L[r,c] * M[r,c,h] * K[r,c],  M=exp(clamp(a*(cs_r-cs_c),0)),
//   K[r,c]=C[r]·B[c],  P[r,c,h]=Σ_d dY[r,h,d]·x[c,h,d],  dcoef=P·dtraw[c]
//   dx[c]=dtraw[c]·Σ_r coef[r,c]·dY[r] ; dK=Σ_h L·M·dcoef ; dB=dKᵀ·C ; dC=dK·B
//   S=coef·dcoef ; dA_log=Σ S·(cs_r-cs_c)·a ; ddt = path1(diretto) + path2(via cumsum)
std::vector<torch::Tensor> mamba2_ssd_bwd(
    torch::Tensor dY,
    torch::Tensor X,
    torch::Tensor dt,
    torch::Tensor A,        // A_log [H]
    torch::Tensor B_param,
    torch::Tensor C_param
) {
    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");

    const auto in_dtype = X.scalar_type();
    const auto comp = (in_dtype == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;
    auto DY = dY.to(comp).contiguous();
    auto XX = X.to(comp).contiguous();
    auto DT = dt.to(comp).contiguous();
    auto AL = A.to(comp).contiguous();
    auto BB = B_param.to(comp).contiguous();
    auto CC = C_param.to(comp).contiguous();

    const int CS = XX.size(2);
    const int H  = AL.size(0);

    auto a    = -torch::exp(AL);                               // [H]
    auto av5  = a.view({1, 1, 1, 1, H});
    auto cs   = DT.cumsum(2);                                  // [B,NC,CS,H]
    auto cs_r = cs.unsqueeze(3);                               // [B,NC,CS,1,H]  (r)
    auto cs_c = cs.unsqueeze(2);                               // [B,NC,1,CS,H]  (c)
    auto M    = torch::exp(torch::clamp_max(av5 * (cs_r - cs_c), 0.0));  // [B,NC,CS,CS,H]
    auto L    = torch::tril(torch::ones({CS, CS}, XX.options())).view({1, 1, CS, CS, 1});
    auto K    = torch::einsum("bnrp,bncp->bnrc", {CC, BB});    // [B,NC,CS,CS]
    auto coef = L * M * K.unsqueeze(-1);                       // [B,NC,CS,CS,H]
    auto P    = torch::einsum("bnrhd,bnchd->bnrch", {DY, XX}); // [B,NC,CS,CS,H]
    auto dcoef = P * DT.unsqueeze(2);                          // dtraw[c] su dim c

    auto dXc = torch::einsum("bnrch,bnrhd->bnchd", {coef, DY});// [B,NC,CS,H,D]
    auto dx  = dXc * DT.unsqueeze(-1);                         // dtraw[c]
    auto ddt_p1 = (coef * P).sum(2);                           // Σ_r coef·P -> [B,NC,CS,H] (c)

    auto dK = (L * M * dcoef).sum(-1);                         // Σ_h -> [B,NC,CS,CS]
    auto dB = torch::einsum("bnrc,bnrp->bncp", {dK, CC});      // [B,NC,CS,P]
    auto dC = torch::einsum("bnrc,bncp->bnrp", {dK, BB});      // [B,NC,CS,P]

    auto S  = coef * dcoef;                                    // [B,NC,CS,CS,H]
    auto dA = (S * (cs_r - cs_c) * av5).sum(c10::IntArrayRef({0, 1, 2, 3}));  // [H]
    auto rowsum = S.sum(3);                                    // Σ_c -> [B,NC,CS,H] (r)
    auto colsum = S.sum(2);                                    // Σ_r -> [B,NC,CS,H] (c)
    auto suf_row = rowsum.flip({2}).cumsum(2).flip({2});       // suffix-sum su r
    auto suf_col = colsum.flip({2}).cumsum(2).flip({2});       // suffix-sum su c
    auto ddt_p2 = a.view({1, 1, 1, H}) * (suf_row - suf_col);
    auto ddt = ddt_p1 + ddt_p2;

    return { dx.to(in_dtype), ddt.to(in_dtype), dA.to(in_dtype),
             dB.to(in_dtype), dC.to(in_dtype) };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mamba2_ssd_fwd, "Mamba-2 SSD Forward (Fused)");
    m.def("backward", &mamba2_ssd_bwd, "Mamba-2 SSD Backward (Fused Recompute)");
}
