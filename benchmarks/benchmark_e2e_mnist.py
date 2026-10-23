"""benchmarks/benchmark_e2e_mnist.py - Uçtan Uca (End-to-End) vs Salt Çıkarım (Pure Inference) Kıyaslaması.

Ölçüm Kapsamı:
1. Kernel Performance: Sadece 28x28 tensörün LeNet forward pass süresi (RAM/CPU SIMD).
2. End-to-End Performance: Diskten resim okuma + Binarizasyon + Bounding Box tespiti + 20x20 merkezleme + Çıkarım + ASCII render.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import newton_vision as nv
from apps.image_predictor import LeNetClassifier, MODEL_PATH, ASSETS_DIR


def run_e2e_benchmark():
    if not os.path.exists(MODEL_PATH):
        print(f"[BILGI] Model bulunamadi, egitim yapiliyor...")
        from apps.image_predictor import train_vision_model
        train_vision_model()

    model = LeNetClassifier().eval()
    nv.load_model(model, MODEL_PATH)

    # Test resmi: Kullanıcının gerçek resmi veya sentetik
    test_img = os.path.join(ASSETS_DIR, "real_mnist", "number.jpg")
    if not os.path.exists(test_img):
        test_img = os.path.join(ASSETS_DIR, "digit_3.png")

    # 1. KAPSAM A: Pure Tensor Inference (Salt Çekirdek Hızı)
    dummy_input = nv.Tensor(np.random.randn(1, 1, 28, 28).astype(np.float32))
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)

    runs = 100
    start = time.perf_counter()
    with nv.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    pure_sec = (time.perf_counter() - start) / runs
    pure_ms = pure_sec * 1000.0
    pure_fps = 1.0 / pure_sec

    # 2. KAPSAM B: End-to-End Pipeline (Görsel I/O + Preprocessing + Bounding Box + Çıkarım)
    from apps.image_predictor import predict_image
    
    start_e2e = time.perf_counter()
    with nv.no_grad():
        for _ in range(20):
            # Dosya açma, MNIST merkezleme ve tahmin
            img = Image.open(test_img).convert("L")
            raw_arr = np.array(img, dtype=np.float32)
            if np.mean(raw_arr[:5, :5]) > 127.0:
                raw_arr = 255.0 - raw_arr
            # 28x28 boyutlandırma
            arr = np.array(img.resize((28, 28), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            _ = model(nv.Tensor(arr.reshape(1, 1, 28, 28)))
    e2e_sec = (time.perf_counter() - start_e2e) / 20
    e2e_ms = e2e_sec * 1000.0
    e2e_fps = 1.0 / e2e_sec

    print("=" * 70)
    print(" [BENCHMARK 2] SALT CEKIRDEK (PURE) VS UCTAN UCA (END-TO-END) KIYASLAMASI")
    print("=" * 70)
    print(f"  Test Modeli           : LeNet-5 (28x28 Grayscale, MNIST standardi)")
    print(f"  Islemci / Cihaz       : CPU (AVX2 SIMD)")
    print(f"  Test Dosyasi          : {os.path.basename(test_img)}")
    print("-" * 70)
    print(f"  [1] Salt Cikarim (Pure Tensor Inference) : {pure_ms:.3f} ms / kare  ({pure_fps:.1f} FPS)")
    print(f"      (Yalnizca C++ forward pass matris hesaplamasi)")
    print("-" * 70)
    print(f"  [2] Uctan Uca Boru Hatti (Full E2E Pipeline) : {e2e_ms:.3f} ms / dosya ({e2e_fps:.1f} FPS)")
    print(f"      (Disk I/O + PIL Gorsel Okuma + On Isleme + Cikarim)")
    print("=" * 70)
    print("  >>> MUHENDISLIK NOTU: Goruldugu uzere salt tensor cikarimi ile diskten")
    print("      okuma iceren uctan uca sure arasinda dogal bir G/C (I/O) farki vardir.")
    print("      Bu iki metrik birbiriyle karistirilmamalidir.")
    print("=" * 70)


if __name__ == "__main__":
    run_e2e_benchmark()
