"""benchmarks/run_all.py - Tek komutla doğrulama ve benchmark süiti.

Kullanım:
    python benchmarks/run_all.py            # parite + uçtan uca LeNet (hızlı)
    python benchmarks/run_all.py --full     # + PyTorch model karşılaştırması (birkaç dakika)

Çalıştırılan scriptler:
1. benchmarks/verify_parity_matrix.py  -> benchmarks/VERIFICATION_REPORT.md (85 noktalı PyTorch paritesi)
2. apps/pytorch_int8_benchmark.py      -> 4 katmanlı CNN paritesi + INT8 bellek ölçümü (konsol)
3. benchmarks/benchmark_e2e_mnist.py   -> salt çıkarım vs uçtan uca LeNet (konsol)
4. (--full) benchmarks/bench_vs_pytorch.py -> benchmarks/BENCH_VS_PYTORCH.md

Herhangi bir script başarısız olursa süit sıfırdan farklı çıkış koduyla biter.
Elle derlenen özet dokümanı: docs/performance.md (otomatik güncellenmez).
"""

from __future__ import annotations
import os
import platform
import subprocess
import sys

import newton_vision as nv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(script: str) -> bool:
    print(f">>> {script}")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    res = subprocess.run([sys.executable, os.path.join(ROOT, script)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[HATA] {script} başarısız (çıkış kodu {res.returncode}):\n{res.stderr}")
        return False
    return True


def main() -> int:
    print("=" * 72)
    print(" [NEWTON-VISION] DOĞRULAMA VE BENCHMARK SÜİTİ")
    print("=" * 72)
    print(f"  * Framework : v{nv.__version__} | Python {platform.python_version()} | {platform.processor() or platform.machine()}")
    print(f"  * AVX2: {nv._core.has_avx2()} | CUDA: {nv._core.has_cuda()}")
    print("=" * 72 + "\n")

    scripts = ["benchmarks/verify_parity_matrix.py", "apps/pytorch_int8_benchmark.py", "benchmarks/benchmark_e2e_mnist.py"]
    if "--full" in sys.argv:
        scripts.append("benchmarks/bench_vs_pytorch.py")
    failed = [s for s in scripts if not run(s)]

    print("=" * 72)
    if failed:
        print(f" [BAŞARISIZ] {len(failed)} script: {', '.join(failed)}")
        return 1
    print(f" [TAMAMLANDI] {len(scripts)} script başarılı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
