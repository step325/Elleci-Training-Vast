"""
Entry-point unico: esegue tutta la suite di test dei kernel custom.

  python run_all_kernel_tests.py                # correttezza + perf + mamba2
  python run_all_kernel_tests.py --sanitizer    # aggiunge compute-sanitizer memcheck

I test sono SOLO additivi: non modificano i sorgenti dei kernel. Servono a
(1) verificare la correttezza su shape allineate e di bordo, (2) misurare
performance/efficienza, (3) far emergere empiricamente i bug noti.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _section(title):
    print("\n\n" + "#" * 92, flush=True)
    print(f"# {title}", flush=True)
    print("#" * 92, flush=True)


def run_sanitizer():
    tool = shutil.which("compute-sanitizer")
    if not tool:
        print("compute-sanitizer non trovato sul PATH — salto il check atomici (B1/B2/B17).")
        return
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))
    cmd = [tool, "--tool", "memcheck", "--error-exitcode", "1",
           sys.executable, os.path.join(HERE, "sanitizer_probe.py")]
    print("$ " + " ".join(cmd[:3]) + " python sanitizer_probe.py\n")
    proc = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    # mostra solo le righe rilevanti del report
    for line in out.splitlines():
        if any(t in line for t in ("ERROR", "Invalid", "Misaligned", "misaligned",
                                   "out of bounds", "========= ", "RESULTS", "hysteresis_v2 OK")):
            print(line)
    print(f"\ncompute-sanitizer exit-code: {proc.returncode} "
          f"(!=0 => accessi illegali rilevati, conferma B1/B2/B17)")


def main():
    do_sanitizer = "--sanitizer" in sys.argv
    env = dict(os.environ, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""))

    for title, mod in (("CORRETTEZZA (griglia shape)", "test_correctness"),
                       ("PERFORMANCE + EFFICIENZA", "test_perf"),
                       ("MAMBA-2 SSD", "test_mamba2"),
                       ("MAMBA-2 GRADIENTI (gradcheck fp64)", "test_mamba2_grad")):
        _section(title)
        proc = subprocess.run([sys.executable, "-m", mod], cwd=HERE, env=env, text=True)
        if proc.returncode != 0:
            print(f"[!] {mod} terminato con codice {proc.returncode}")

    if do_sanitizer:
        _section("COMPUTE-SANITIZER — atomici pack INT2 (packed_K % 4 != 0)")
        run_sanitizer()


if __name__ == "__main__":
    main()
