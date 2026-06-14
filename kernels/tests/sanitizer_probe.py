"""
Probe per compute-sanitizer: esegue hysteresis_v2 con packed_K NON multiplo di 4
(K=20 -> packed_K=5) per far emergere atomici disallineati / OOB in
pack_int2 / pack_int4 (bug B1 / B2 / B17).

Eseguire con:  compute-sanitizer --tool memcheck python sanitizer_probe.py
"""
import torch

from kernel_build import build_int2, make_packed_weights

ops = build_int2()
N, K = 64, 20  # packed_K = 5 (5 % 4 = 1) -> base riga non allineata a 4 byte
W_packed, _ = make_packed_weights(N, K, "cuda", seed=5)
H_packed = torch.randint(0, 256, (N, (K + 1) // 2), dtype=torch.uint8, device="cuda")
dW = torch.randn(N, K, dtype=torch.float32, device="cuda")

ops.hysteresis_step_v2(dW, W_packed, H_packed, 1e-2, 1.0, 7, 0.99, 1)
torch.cuda.synchronize()
print(f"hysteresis_v2 OK su N={N} K={K} packed_K={(K + 3) // 4} (atteso: report memcheck)")
