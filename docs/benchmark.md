# 📊 Bilimsel Karşılaştırma & Benchmark Metodolojisi

`newton-vision`, performans ölçümlerinde yapay (synthetic) iddialar yerine istatistiksel örnekleme ile elde edilmiş şeffaf verileri sunar.

---

## 💻 Test Çevresi & Donanım Özellikleri

- **İşlemci (CPU):** `Intel64 Family 6 Model 141 Stepping 1 (12 Core / 12 Threads)`
- **Thread Yönetimi:** `OpenMP (OMP_NUM_THREADS=12)`
- **Veri Hassasiyeti:** `FP32 (Single Precision)`
- **Karşılaştırılan Kütüphane:** `PyTorch v2.6.0+cpu`
- **Donanım İvmelendirmesi:** `256-bit AVX2 FMA Vector Engine`
- **İstatistiki Örneklem:** `Warmup=3, Runs=10` (`bench_vs_pytorch.py` içindeki `WARMUP`/`RUNS` sabitleri)
- **Mod:** İki framework de `eval()` + gradyan kapalı (`nv.no_grad()` / `torch.no_grad()`)
- **Adillik kontrolü:** Script, iki modelin parametre sayısı eşleşmezse durur

---

## 📈 İstatistiksel Karşılaştırma Raporu

Ölçüm tarihi 2026-10-23. Ham çıktı: [`benchmarks/BENCH_VS_PYTORCH.md`](../benchmarks/BENCH_VS_PYTORCH.md).

| Model | Framework | Parametre | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | FPS | Karşılaştırma |
|---|---|---|---|---|---|---|---|---|
| **ResNet-18** | newton-vision | 11,689,512 | **122.97** | ±6.09 | 112.91 | 131.35 | **8.1** | **4.19x Yavaş** |
| | PyTorch CPU | 11,689,512 | 29.34 | ±1.39 | 27.42 | 31.80 | 34.1 | Base |
| **MobileNetV2** | newton-vision | 3,504,872 | **109.18** | ±12.57 | 91.22 | 128.67 | **9.2** | **4.34x Yavaş** |
| | PyTorch CPU | 3,504,872 | 25.18 | ±6.76 | 19.78 | 43.84 | 39.7 | Base |
| **ViT-Tiny (Transformer)** | newton-vision | 5,717,416 | **61.25** | ±7.03 | 50.52 | 76.76 | **16.3** | **2.53x Yavaş** |
| | PyTorch CPU (torchvision `VisionTransformer`, aynı boyutlar) | 5,717,416 | 24.18 | ±1.21 | 22.14 | 26.30 | 41.3 | Base |

> **Ölçüm değişkenliği:** Bu dizüstü makinede tekrarlı koşularda mutlak süreler belirgin şekilde değişiyor (ör. ViT-Tiny 58–84 ms; ViT kodu değişmediği halde oranı 2.5–3.7x arasında oynadı).
>
> **C++ optimizasyonunun etkisi (aynı gün, önce → sonra):** MobileNetV2 çıkarımı 1565 ms → 109 ms (PyTorch'a göre ~73x → ~4.3x yavaş). Gruplu/depthwise evrişim artık gruplar üzerinden paralel çalışıyor. ResNet-18 ve ViT'nin forward yolu değişmedi.
>
> **Düzeltme:** Eski "ViT-Tiny 2.28x hızlı" sonucu geçersizdir. Script PyTorch tarafında ViT-B/16 (86M parametre) ölçüyordu ve newton-vision modelini `eval`/`no_grad` olmadan çalıştırıyordu.

---

## 🔬 Bilimsel Analiz & Bulgular

1. **Vision Transformer (ViT-Tiny):** Aynı mimaride PyTorch CPU yaklaşık **2.5–3.7 kat daha hızlıdır** (koşudan koşuya değişken). newton-vision'da Linear/MHSA matris çarpımları NumPy üzerinden, katman orkestrasyonu ise Python'da yürür.
2. **ResNet-18:** PyTorch CPU ~4 kat daha hızlıdır.
3. **MobileNetV2:** Fark ~4.3x'e indi (C++ optimizasyonundan önce ~73x). Depthwise (groups = C) evrişim için hâlâ özel bir çekirdek yok; genel gruplu im2col yolu gruplar üzerinden paralel çalışıyor. PyTorch'un burada hangi algoritmayı seçtiği bu repoda ölçülmemiştir.
