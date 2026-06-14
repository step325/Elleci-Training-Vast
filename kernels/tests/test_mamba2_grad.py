"""
Verifica gradienti del kernel Mamba-2 SSD confrontandoli con l'autograd di una
reference PyTorch differenziabile (chunk-local), in DOPPIA precisione.

Stampa il relerr per ciascun gradiente: dx, ddt, dA, dB, dC.
Serve sia a SCOPARE M1 (quali gradienti sono rotti) sia a VERIFICARE il fix.
"""
import torch

from kernel_build import build_mamba2

CS = 64  # CHUNK_SIZE hardcoded nel kernel


def ref_fwd(X, dt, A_log, Bp, Cp):
    """Forward intra-chunk differenziabile, stessa math del kernel, nel dtype di X."""
    Bn, NC, _, H, D = X.shape
    a = -torch.exp(A_log)  # [H]
    L = torch.tril(torch.ones(CS, CS, dtype=X.dtype, device=X.device))
    outs = []
    for bb in range(Bn):
        for cc in range(NC):
            for h in range(H):
                dtc = dt[bb, cc, :, h]
                cs = dtc.cumsum(0)
                decay = torch.exp(torch.clamp(a[h] * (cs[:, None] - cs[None, :]), max=0.0))
                Kmat = Cp[bb, cc] @ Bp[bb, cc].t()           # [CS,CS] = C[r]·B[c]
                coef = L * decay * Kmat
                xdt = X[bb, cc, :, h, :] * dtc[:, None]       # [CS,D]
                outs.append(coef @ xdt)                       # [CS,D]
    # outs in ordine (bb,cc,h) -> ricostruisci [Bn,NC,CS,H,D]
    Y = torch.stack(outs, 0).reshape(Bn, NC, H, CS, D).permute(0, 1, 3, 2, 4).contiguous()
    return Y


def make_fn(ops):
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, dt, A, B, C):
            ctx.save_for_backward(x, dt, A, B, C)
            return ops.forward(x.contiguous(), dt.contiguous(), A.contiguous(),
                               B.contiguous(), C.contiguous())

        @staticmethod
        def backward(ctx, gy):
            x, dt, A, B, C = ctx.saved_tensors
            dx, ddt, dA, dB, dC = ops.backward(gy.contiguous(), x, dt, A, B, C)
            return dx, ddt, dA, dB, dC
    return Fn


def _grads(fwd, inits, g):
    xs = [t.clone().requires_grad_(True) for t in inits]
    Y = fwd(*xs)
    Y.backward(g)
    return [t.grad for t in xs]


def main():
    ops = build_mamba2()
    Fn = make_fn(ops)
    dev, dt_ = "cuda", torch.double
    Bn, NC, H, D, P = 1, 1, 2, 4, 4
    torch.manual_seed(0)
    x0 = torch.randn(Bn, NC, CS, H, D, dtype=dt_, device=dev)
    dt0 = torch.rand(Bn, NC, CS, H, dtype=dt_, device=dev) * 0.1 + 0.02
    A0 = torch.randn(H, dtype=dt_, device=dev)
    B0 = torch.randn(Bn, NC, CS, P, dtype=dt_, device=dev) * 0.5
    C0 = torch.randn(Bn, NC, CS, P, dtype=dt_, device=dev) * 0.5
    g = torch.randn(Bn, NC, CS, H, D, dtype=dt_, device=dev)
    inits = (x0, dt0, A0, B0, C0)

    # sanity: forward kernel vs reference
    yk = ops.forward(x0, dt0, A0, B0, C0)
    yr = ref_fwd(x0, dt0, A0, B0, C0)
    print(f"forward relerr: {(yk - yr).norm().item() / yr.norm().item():.2e}")

    ref = _grads(ref_fwd, inits, g)
    ker = _grads(lambda x, dt, A, B, C: Fn.apply(x, dt, A, B, C), inits, g)

    print(f"\n{'grad':<5}{'relerr':>12}{'kernel_norm':>14}{'ref_norm':>12}   esito")
    print("-" * 56)
    ok_all = True
    for n, r, k in zip(["dx", "ddt", "dA", "dB", "dC"], ref, ker):
        rel = (k - r).norm().item() / (r.norm().item() + 1e-15)
        ok = rel < 1e-6
        ok_all = ok_all and ok
        print(f"{n:<5}{rel:>12.2e}{k.norm().item():>14.4e}{r.norm().item():>12.4e}   "
              f"{'OK' if ok else 'FAIL'}")
    print("-" * 56)
    print("TUTTI OK" if ok_all else "ALCUNI GRADIENTI ERRATI")


if __name__ == "__main__":
    main()
