"""
Orchestratore correttezza INT2: esegue una griglia di shape (allineate + di
bordo) lanciando `correctness_worker.py` in subprocess isolati.

Le shape di bordo (N%8!=0, K%16!=0, K%4!=0) sono scelte apposta per innescare i
bug di allineamento dei load/store vettoriali. Un fault CUDA su una shape NON
contamina le altre grazie all'isolamento per-processo.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# (kind, M, N, K, atteso_pass, nota/bug-id)
CASES = [
    # --- baseline allineate: devono passare ---
    ("matmul_fp16",   256, 256, 256, True,  "aligned"),
    ("matmul_int8",  4096, 4096, 2048, True, "aligned big (TC path)"),
    ("matmul_int8",   256, 256, 256, True,  "aligned (scalar/med route)"),
    ("backward_fp16", 256, 256, 256, True,  "aligned"),
    ("backward_int8", 4096, 4096, 2048, True, "aligned big (TC path)"),
    ("unpack",        256, 256, 256, True,  "aligned"),
    ("quantize",      256, 256, 256, True,  "aligned"),
    ("matmul_fp16",     8,   8,   8, True,  "small (scalar fallback)"),
    ("matmul_int8",     8,   8,   8, True,  "small (scalar fallback)"),
    # --- bordo: N non multiplo di 8 -> ora gestito da fallback scalare (B5/B9 FIXED) ---
    ("matmul_fp16",   256, 260, 256, True,  "B5 FIXED N%8!=0 (fallback scalare)"),
    ("matmul_int8",  4096, 4100, 2048, True, "B9 FIXED N%8!=0 (fallback scalare)"),
    # --- bordo: K non multiplo di 16 -> int4/uint32 load gated (B9 FIXED) ---
    ("matmul_int8",  4096, 4096, 2056, True, "B9 FIXED K%16!=0 (load gated)"),
    ("backward_int8", 4096, 4100, 2048, True, "B14 FIXED N%16/K%8 (gated)"),
    # --- bordo: K dX non multiplo di 8 -> float4 store dX gated (B11 FIXED) ---
    ("backward_fp16", 256, 256, 260, True,  "B11 FIXED K%8!=0 (store gated)"),
    # --- bordo: K non multiplo di 4 -> K ricavato da X (B26 FIXED) ---
    ("matmul_fp16",   128, 128, 130, True,  "B26 FIXED K%4!=0 (K da X.size(-1))"),
    ("backward_fp16", 128, 128, 130, True,  "backward K%4!=0 K esplicito (ora regge, B11 fix)"),
    # --- unpack su shape irregolari (riferimento corretto) ---
    ("unpack",        130, 126, 126, True,  "unpack shape irregolare"),
    ("quantize",        1, 1, 255, True,    "quantize size%4!=0 (remainder loop)"),
]


def run_case(kind, M, N, K):
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "correctness_worker.py"), kind, str(M), str(N), str(K)],
        cwd=HERE, env=env, capture_output=True, text=True, timeout=600,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"status": "error", "error": (proc.stderr.strip()[-300:] or "no output")}


def verdict(res, expect_pass):
    st = res.get("status")
    if st == "pass":
        ok = res.get("relerr", 9e9) <= res.get("tol", 0)
        real = "PASS" if ok else "MISMATCH"
    elif st == "error":
        real = "CUDA_ERR"
    else:
        real = st.upper()
    passed = (real == "PASS")
    flag = "  " if passed == expect_pass else "⚠️"
    return real, flag


def main():
    print("=" * 92)
    print("CORRETTEZZA KERNEL INT2 — griglia shape (allineate + bordo)")
    print("=" * 92)
    print(f"{'kind':<15}{'M':>5}{'N':>6}{'K':>6}  {'atteso':<8}{'reale':<10}{'relerr':>11}  nota")
    print("-" * 92)
    results = []
    for kind, M, N, K, expect, note in CASES:
        res = run_case(kind, M, N, K)
        real, flag = verdict(res, expect)
        relerr = res.get("relerr")
        relerr_s = f"{relerr:.3e}" if isinstance(relerr, (int, float)) else "-"
        exp_s = "PASS" if expect else "FAIL/ERR"
        print(f"{kind:<15}{M:>5}{N:>6}{K:>6}  {exp_s:<8}{real:<10}{relerr_s:>11} {flag} {note}")
        results.append((kind, M, N, K, real, expect, note, res))
    print("-" * 92)
    surprises = [r for r in results if (r[4] == "PASS") != r[5]]
    print(f"Casi: {len(results)} | conformi all'attesa: {len(results)-len(surprises)} | "
          f"divergenti (⚠️): {len(surprises)}")
    print("Nota: 'FAIL/ERR' atteso = il bug di allineamento si manifesta come "
          "CUDA_ERR (cudaErrorMisalignedAddress) o MISMATCH.")
    return results


if __name__ == "__main__":
    main()
