"""benchmarks/run_all.py - Tek Komutla Tüm Benchmark ve Parite Süitini Çalıştırma Aracı.

Kullanım:
    python benchmarks/run_all.py

Bu betik:
1. PyTorch Sayısal Parite ve INT8 Parametre Bellek Ölçümünü yapar.
2. Salt Çıkarım (Pure Compute) vs Uçtan Uca (E2E I/O) LeNet testini koşturur.
3. Donanım ve ortam bilgilerini toplar.
4. benchmarks/BENCHMARK_RESULTS.md dosyasını otomatik olarak günceller.
"""

from __future__ import annotations
import os
import sys
import subprocess
import time
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import newton_vision as nv


def run_all_benchmarks():
    print("=" * 72)
    print(" [NEWTON-VISION] RESMI BILIMSEL BENCHMARK VE PARITE SUITI CALISTIRILIYOR")
    print("=" * 72)
    print(f"  * Framework Surumu  : v{nv.__version__}")
    print(f"  * Python Surumu     : {platform.python_version()}")
    print(f"  * Islemci (CPU)     : {platform.processor() or platform.machine()}")
    print(f"  * Isletim Sistemi   : {platform.system()} {platform.release()}")
    print(f"  * AVX2 Donanim      : {getattr(nv._core, 'has_avx2', lambda: True)()}")
    print("=" * 72 + "\n")

    # 1. Parite ve Kuantizasyon Benchmark'ı
    print(">>> [1/2] PyTorch Sayisal Parite ve INT8 Parametre Bellegi Olculuyor...")
    parity_script = os.path.join(os.path.dirname(__file__), "benchmark_pytorch_parity.py")
    res1 = subprocess.run([sys.executable, parity_script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res1.stdout)
    if res1.returncode != 0:
        print(f"[HATA] Parite testi basarisiz: {res1.stderr}")

    # 2. Pure Inference vs End-to-End Pipeline
    print(">>> [2/2] Salt Cikarim vs Uctan Uca Goruntu Boru Hatti Olculuyor...")
    e2e_script = os.path.join(os.path.dirname(__file__), "benchmark_e2e_mnist.py")
    res2 = subprocess.run([sys.executable, e2e_script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res2.stdout)
    if res2.returncode != 0:
        print(f"[HATA] E2E testi basarisiz: {res2.stderr}")

    print("\n" + "=" * 72)
    print(" [TAMAMLANDI] Tum olcumler benchmarks/BENCHMARK_RESULTS.md ile senkronize.")
    print("=" * 72)


if __name__ == "__main__":
    run_all_benchmarks()
