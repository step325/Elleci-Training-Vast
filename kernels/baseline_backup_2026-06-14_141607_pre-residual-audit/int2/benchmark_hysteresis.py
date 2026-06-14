import torch
import torch.utils.benchmark as benchmark
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from core.bitnet.int2_linear_tc import int2_tc_ops

def run_benchmark():
    N = 4096
    K = 4096
    
    dW = torch.randn(N, K, dtype=torch.float32, device='cuda')
    packed_K = (K + 3) // 4
    packed_H = (K + 1) // 2
    
    W_packed = torch.randint(0, 256, (N, packed_K), dtype=torch.uint8, device='cuda')
    H_packed = torch.randint(0, 256, (N, packed_H), dtype=torch.uint8, device='cuda')
    
    t_v2 = benchmark.Timer(
        stmt='int2_tc_ops.hysteresis_step_v2(dW, W_packed, H_packed, 1e-4, 1.0, 7, 0.99, 1)',
        globals={'int2_tc_ops': int2_tc_ops, 'dW': dW, 'W_packed': W_packed, 'H_packed': H_packed}
    )
    
    res = t_v2.timeit(100)
    print(f"Hysteresis V2 (Atomics/Shuffle): {res.mean * 1000:.3f} ms")

if __name__ == "__main__":
    run_benchmark()
