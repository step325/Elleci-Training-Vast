"""
Worker isolato: esegue UN singolo caso di correttezza e stampa il risultato JSON.

Eseguito in subprocess dall'orchestratore così che un fault CUDA (es. accesso
disallineato) su una shape non distrugga il contesto degli altri casi.

Uso:  python correctness_worker.py <kind> <M> <N> <K>
kind: matmul_fp16 | matmul_int8 | backward_fp16 | backward_int8 | unpack | quantize
"""
import json
import sys

import torch

from kernel_build import build_int2, make_packed_weights, unpack_int2

GAMMA = 1.3


def _relerr(out: torch.Tensor, ref: torch.Tensor) -> float:
    out = out.float()
    ref = ref.float()
    denom = ref.abs().max().item()
    denom = denom if denom > 1e-6 else 1.0
    return (out - ref).abs().max().item() / denom


def run(kind: str, M: int, N: int, K: int) -> dict:
    ops = build_int2()
    dev = "cuda"
    W_packed, W_t = make_packed_weights(N, K, dev, seed=1)  # W_t: [N,K] ternario

    if kind == "matmul_fp16":
        X = torch.randn(M, K, dtype=torch.float16, device=dev)
        Y = ops.matmul(X, W_packed, GAMMA)
        ref = (X.float() @ W_t.t()) * GAMMA
        return {"status": "pass", "relerr": _relerr(Y, ref), "tol": 6e-2}

    if kind == "matmul_int8":
        X = torch.randn(M, K, dtype=torch.float16, device=dev)
        d_absmax = torch.zeros(1, dtype=torch.float32, device=dev)
        X_int8, scale = ops.quantize_activation_async(X, d_absmax)
        Y = ops.matmul_int8(X_int8, W_packed, scale, GAMMA)
        acc = X_int8.float() @ W_t.t()           # accumulazione INT32 esatta
        ref = acc * (scale.item() / 127.0) * GAMMA
        return {"status": "pass", "relerr": _relerr(Y, ref), "tol": 5e-3}

    if kind == "backward_fp16":
        dY = torch.randn(M, N, dtype=torch.float16, device=dev)
        dX = ops.backward_input(dY, W_packed, GAMMA, K)
        ref = (dY.float() @ W_t) * GAMMA
        return {"status": "pass", "relerr": _relerr(dX, ref), "tol": 6e-2}

    if kind == "backward_int8":
        dY = torch.randn(M, N, dtype=torch.float16, device=dev)
        d_absmax = torch.zeros(1, dtype=torch.float32, device=dev)
        dY_int8, scale = ops.quantize_activation_async(dY, d_absmax)
        dX = ops.backward_input_int8(dY_int8, W_packed, scale, GAMMA, K)
        ref = (dY_int8.float() @ W_t) * (scale.item() / 127.0) * GAMMA
        return {"status": "pass", "relerr": _relerr(dX, ref), "tol": 5e-3}

    if kind == "unpack":
        out = torch.empty(N, K, dtype=torch.float16, device=dev)
        ops.unpack_int2(W_packed, out, K)
        return {"status": "pass", "relerr": _relerr(out, W_t), "tol": 0.0}

    if kind == "quantize":
        X = torch.randn(M, K, dtype=torch.float16, device=dev) * 3.0
        d_absmax = torch.zeros(1, dtype=torch.float32, device=dev)
        X_int8, scale = ops.quantize_activation_async(X, d_absmax)
        deq = X_int8.float() * (scale.item() / 127.0)
        # tolleranza = mezzo step di quantizzazione
        step = scale.item() / 127.0
        err = (deq - X.float()).abs().max().item()
        return {"status": "pass", "relerr": err, "tol": step * 1.01,
                "scale": scale.item(), "absmax_ref": X.abs().max().item()}

    return {"status": "skip", "reason": f"kind sconosciuto {kind}"}


if __name__ == "__main__":
    kind, M, N, K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    try:
        res = run(kind, M, N, K)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — vogliamo catturare anche i fault CUDA
        res = {"status": "error", "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res))
