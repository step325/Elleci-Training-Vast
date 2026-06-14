"""
Performance + efficienza dei kernel INT2 (in-process, shape allineate).

Misura latenza (CUDA events), throughput (TFLOPS effettivi = 2*M*N*K/t) e
picco VRAM per ogni kernel, confrontando con la baseline cuBLAS FP16.
"""
import torch

from kernel_build import build_int2, make_packed_weights

GAMMA = 1.3
# shape rappresentative: (M = batch*seq, K = in_features, N = out_features)
SHAPES = [
    (4096, 2048, 2048),   # tipica LLM (come benchmark esistente)
    (2048, 1536, 1536),   # RTX 4070 config (d_model 1536)
]


def _warmup_gpu():
    """Ramp dei clock GPU + init handle cuBLAS prima dei bench.

    Senza questo la prima matmul cuBLAS misurata paga clock non rampati e
    algo-select della prima shape: baseline falsato (es. 7.5 invece di ~54
    TFLOPS su RTX 4070). Va chiamato una volta prima di tutti i timing.
    """
    a = torch.randn(8192, 8192, dtype=torch.float16, device="cuda")
    b = torch.randn(8192, 8192, dtype=torch.float16, device="cuda")
    for _ in range(50):
        torch.matmul(a, b)  # risultato scartato: serve solo a rampare i clock
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _peak_vram_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def _tflops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def bench_matmul(ops, M, K, N):
    X = torch.randn(M, K, dtype=torch.float16, device="cuda")
    dY = torch.randn(M, N, dtype=torch.float16, device="cuda")
    W_packed, W_t = make_packed_weights(N, K, "cuda", seed=2)
    W_fp16 = W_t.half()
    d_absmax = torch.zeros(1, dtype=torch.float32, device="cuda")
    X_int8, scale_x = ops.quantize_activation_async(X, d_absmax)
    dY_int8, scale_dy = ops.quantize_activation_async(dY, d_absmax)

    rows = []
    rows.append(("FWD cuBLAS fp16", _time_ms(lambda: torch.matmul(X, W_fp16.t())), M, N, K))
    rows.append(("FWD int2 v1 (WMMA fp16)", _time_ms(lambda: ops.matmul(X, W_packed, GAMMA)), M, N, K))
    rows.append(("FWD int2 v2 (IMMA int8)", _time_ms(lambda: ops.matmul_int8(X_int8, W_packed, scale_x, GAMMA)), M, N, K))
    rows.append(("BWD cuBLAS fp16", _time_ms(lambda: torch.matmul(dY, W_fp16)), M, N, K))
    rows.append(("BWD int2 v1 (WMMA fp16)", _time_ms(lambda: ops.backward_input(dY, W_packed, GAMMA, K)), M, N, K))
    rows.append(("BWD int2 v2 (IMMA int8)", _time_ms(lambda: ops.backward_input_int8(dY_int8, W_packed, scale_dy, GAMMA, K)), M, N, K))
    return rows


def bench_aux(ops):
    """Hysteresis / quantize / unpack su una shape pesi tipica."""
    N, K = 4096, 4096
    packed_K, packed_H = (K + 3) // 4, (K + 1) // 2
    W_packed, _ = make_packed_weights(N, K, "cuda", seed=3)
    H_packed = torch.randint(0, 256, (N, packed_H), dtype=torch.uint8, device="cuda")
    dW = torch.randn(N, K, dtype=torch.float32, device="cuda")
    X = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    d_absmax = torch.zeros(1, dtype=torch.float32, device="cuda")
    W_fp16_out = torch.empty(N, K, dtype=torch.float16, device="cuda")
    X_int8, scale = ops.quantize_activation_async(X, d_absmax)

    print(f"\n  hysteresis_v2 [{N}x{K}]   : {_time_ms(lambda: ops.hysteresis_step_v2(dW, W_packed, H_packed, 1e-4, 1.0, 7, 0.99, 1)):.4f} ms")
    print(f"  quantize_activation [{X.numel()//10**6}M]: {_time_ms(lambda: ops.quantize_activation_async(X, d_absmax)):.4f} ms")
    print(f"  dequantize_activation     : {_time_ms(lambda: ops.dequantize_activation(X_int8, scale)):.4f} ms")
    print(f"  unpack_int2 [{N}x{K}]      : {_time_ms(lambda: ops.unpack_int2(W_packed, W_fp16_out, K)):.4f} ms")
    print(f"  VRAM unpack buffer fp16   : {_peak_vram_mb(lambda: ops.unpack_int2(W_packed, W_fp16_out, K)):.1f} MB")


def main():
    ops = build_int2()
    _warmup_gpu()
    print("=" * 84)
    print("PERFORMANCE + EFFICIENZA KERNEL INT2")
    print("Nota: clock GPU rampato prima dei bench. INT2 v2 pareggia cuBLAS fp16")
    print("sul compute (~50 TFLOPS su 4070); il vantaggio INT2 e' la VRAM, non lo speed.")
    print("=" * 84)
    for (M, K, N) in SHAPES:
        print(f"\nShape M={M} K={K} N={N}")
        print(f"  {'kernel':<28}{'ms':>10}{'TFLOPS':>10}")
        print("  " + "-" * 48)
        for name, ms, m, n, k in bench_matmul(ops, M, K, N):
            print(f"  {name:<28}{ms:>10.4f}{_tflops(m, n, k, ms):>10.1f}")
    bench_aux(ops)
    print("=" * 84)


if __name__ == "__main__":
    main()
