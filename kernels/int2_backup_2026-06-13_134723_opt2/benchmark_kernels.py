# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.utils.benchmark as benchmark
import sys
from pathlib import Path

# Aggiungi il root al sys.path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from core.bitnet.int2_linear_tc import int2_tc_ops, HAS_TC_OPS

if not HAS_TC_OPS:
    print("TC Ops not loaded.")
    sys.exit(1)

def run_benchmark():
    # Dimensioni tipiche da LLM
    M = 4096  # Batch * Seq
    K = 2048  # In features
    N = 2048  # Out features

    print(f"Benchmarking with dimensions: M={M}, K={K}, N={N}")
    print("-" * 60)

    # Dati FP16
    X = torch.randn(M, K, dtype=torch.float16, device='cuda')
    dY = torch.randn(M, N, dtype=torch.float16, device='cuda')
    
    # Baseline FP16 Weights per torch.matmul
    W_fp16 = torch.randn(N, K, dtype=torch.float16, device='cuda')

    # Dati INT2 packed
    packed_K = (K + 3) // 4
    W_packed = torch.randint(0, 256, (N, packed_K), dtype=torch.uint8, device='cuda')
    gamma = 1.5
    
    # Preparazione per v2 (INT8)
    d_absmax = torch.zeros(1, dtype=torch.float32, device='cuda')
    X_int8, scale_x = int2_tc_ops.quantize_activation_async(X, d_absmax)
    dY_int8, scale_dy = int2_tc_ops.quantize_activation_async(dY, d_absmax)

    # 1. Forward Matmul
    results_fwd = []
    
    # Baseline cuBLAS FP16
    t_fp16 = benchmark.Timer(
        stmt='torch.matmul(X, W_fp16.t())',
        globals={'X': X, 'W_fp16': W_fp16, 'torch': torch}
    )
    results_fwd.append(t_fp16.timeit(100))
    
    # V1 (FP16 WMMA)
    t_v1 = benchmark.Timer(
        stmt='int2_tc_ops.matmul(X, W_packed, gamma)',
        globals={'int2_tc_ops': int2_tc_ops, 'X': X, 'W_packed': W_packed, 'gamma': gamma}
    )
    results_fwd.append(t_v1.timeit(100))
    
    # V2 (INT8 IMMA)
    t_v2 = benchmark.Timer(
        stmt='int2_tc_ops.matmul_int8(X_int8, W_packed, scale_x, gamma)',
        globals={'int2_tc_ops': int2_tc_ops, 'X_int8': X_int8, 'W_packed': W_packed, 'scale_x': scale_x, 'gamma': gamma}
    )
    results_fwd.append(t_v2.timeit(100))

    # 2. Backward dX
    results_bwd = []
    
    # Baseline cuBLAS FP16
    t_fp16_bwd = benchmark.Timer(
        stmt='torch.matmul(dY, W_fp16)',
        globals={'dY': dY, 'W_fp16': W_fp16, 'torch': torch}
    )
    results_bwd.append(t_fp16_bwd.timeit(100))
    
    # V1 Backward
    t_v1_bwd = benchmark.Timer(
        stmt='int2_tc_ops.backward_input(dY, W_packed, gamma, K)',
        globals={'int2_tc_ops': int2_tc_ops, 'dY': dY, 'W_packed': W_packed, 'gamma': gamma, 'K': K}
    )
    results_bwd.append(t_v1_bwd.timeit(100))
    
    # V2 Backward
    t_v2_bwd = benchmark.Timer(
        stmt='int2_tc_ops.backward_input_int8(dY_int8, W_packed, scale_dy, gamma, K)',
        globals={'int2_tc_ops': int2_tc_ops, 'dY_int8': dY_int8, 'W_packed': W_packed, 'scale_dy': scale_dy, 'gamma': gamma, 'K': K}
    )
    results_bwd.append(t_v2_bwd.timeit(100))

    # Stampo i risultati
    print("FORWARD PASS (Y = X @ W^T)")
    print(f"FP16 cuBLAS baseline: {results_fwd[0].mean * 1000:.3f} ms")
    print(f"INT2 v1 (FP16 WMMA):  {results_fwd[1].mean * 1000:.3f} ms")
    print(f"INT2 v2 (INT8 IMMA):  {results_fwd[2].mean * 1000:.3f} ms")
    print("")
    print("BACKWARD PASS (dX = dY @ W)")
    print(f"FP16 cuBLAS baseline: {results_bwd[0].mean * 1000:.3f} ms")
    print(f"INT2 v1 (FP16 WMMA):  {results_bwd[1].mean * 1000:.3f} ms")
    print(f"INT2 v2 (INT8 IMMA):  {results_bwd[2].mean * 1000:.3f} ms")
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
