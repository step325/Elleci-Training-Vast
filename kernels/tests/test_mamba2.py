"""
Test Mamba-2 SSD kernel: correttezza forward (vs reference PyTorch chunk-local),
controllo gradienti backward e performance.

NB: il kernel calcola il contributo INTRA-chunk (blocco diagonale). Il reference
è quindi anch'esso chunk-local. Lo stato inter-chunk non è propagato dal kernel
(vedi nota di audit M3).
"""
import torch

from kernel_build import build_mamba2

CS = 64  # CHUNK_SIZE hardcoded nel kernel


def ref_forward(X, dt, A_log, Bp, Cp):
    Bn, NC, _, H, D = X.shape
    a = -torch.exp(A_log)  # [H]
    Y = torch.zeros_like(X)
    L = torch.tril(torch.ones(CS, CS, device=X.device, dtype=torch.float32))
    for bb in range(Bn):
        for h in range(H):
            for cc in range(NC):
                dtc = dt[bb, cc, :, h].float()
                cs = dtc.cumsum(0)
                decay = torch.exp(torch.clamp(a[h].float() * (cs[:, None] - cs[None, :]), max=0.0))
                Kmat = Cp[bb, cc].float() @ Bp[bb, cc].float().t()  # [CS,CS] = C[r]·B[c]
                coef = L * decay * Kmat
                xdt = X[bb, cc, :, h, :].float() * dtc[:, None]
                Y[bb, cc, :, h, :] = (coef @ xdt).to(Y.dtype)
    return Y


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ops = build_mamba2()
    dev = "cuda"
    Bn, NC, H, D, P = 2, 3, 4, 32, 16
    torch.manual_seed(0)
    X = torch.randn(Bn, NC, CS, H, D, device=dev, dtype=torch.float32)
    dt = torch.rand(Bn, NC, CS, H, device=dev, dtype=torch.float32) * 0.1  # dt>0
    A_log = torch.randn(H, device=dev, dtype=torch.float32)
    Bp = torch.randn(Bn, NC, CS, P, device=dev, dtype=torch.float32) * 0.5
    Cp = torch.randn(Bn, NC, CS, P, device=dev, dtype=torch.float32) * 0.5

    print("=" * 84)
    print("MAMBA-2 SSD KERNEL — correttezza + gradienti + performance")
    print("=" * 84)

    # --- Forward ---
    Y = ops.forward(X, dt, A_log, Bp, Cp)
    Yref = ref_forward(X, dt, A_log, Bp, Cp)
    rel = (Y - Yref).abs().max().item() / max(Yref.abs().max().item(), 1e-6)
    print(f"\nForward intra-chunk relerr vs reference: {rel:.3e}  "
          f"-> {'PASS' if rel < 2e-3 else 'MISMATCH'}")

    # --- Backward: il kernel calcola solo dX e ddt; dA/dB/dC restano a zero ---
    dY = torch.randn_like(X)
    dX, ddt, dA, dB, dC = ops.backward(dY, X, dt, A_log, Bp, Cp)
    print("\nBackward — controllo gradienti restituiti:")
    print(f"  dX  : finite={torch.isfinite(dX).all().item()}  max|.|={dX.abs().max().item():.3e}")
    print(f"  ddt : finite={torch.isfinite(ddt).all().item()}  max|.|={ddt.abs().max().item():.3e}")
    for nm, g in (("dA", dA), ("dB", dB), ("dC", dC)):
        nz = g.abs().max().item() > 0.0
        flag = "ok (M1 risolto)" if nz else "⚠️ ZERO inatteso (regressione M1)"
        print(f"  {nm}  : max|.|={g.abs().max().item():.3e}  {flag}")

    # --- Performance ---
    print(f"\nPerformance (B={Bn} NC={NC} CS={CS} H={H} D={D} P={P}):")
    print(f"  forward : {_time_ms(lambda: ops.forward(X, dt, A_log, Bp, Cp)):.4f} ms")
    print(f"  backward: {_time_ms(lambda: ops.backward(dY, X, dt, A_log, Bp, Cp)):.4f} ms")
    print("=" * 84)


if __name__ == "__main__":
    main()
