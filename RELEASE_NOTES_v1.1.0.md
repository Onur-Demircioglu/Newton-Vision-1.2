# 🚀 newton-vision v1.1.0 Release Notes — Maintainer & Scientific Quality Release

> **Sürüm Tarihi:** 23 Temmuz 2026  
> **PyPI Adresi:** [https://pypi.org/project/newton-vision/1.1.0/](https://pypi.org/project/newton-vision/)

`newton-vision v1.1.0`, projenin deneysel bir kod deposundan **maintainer disipliniyle yönetilen, şeffaf, dokümante edilmiş ve güvenilir bir açık kaynak framework'üne** dönüştüğü resmi sürümdür.

---

## 🌟 Öne Çıkan Değişiklikler ve Yenilikler

### 1. 📊 Bilimsel Standartlarda Benchmark Paketi (`benchmarks/`)
- `benchmarks/bench_vs_pytorch.py` ile `mean`, `std`, `min`, `max` istatistiksel örneklemesi (`Warmup=5, Runs=20`).
- **Vision Transformer (ViT-Tiny):** `newton-vision` (210.62 ms ± 18.33 ms), PyTorch v2.6 CPU'ya (479.76 ms ± 29.48 ms) karşı **2.28 KAT DAHA HIZLI** tescillendi.
- [BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md) ve [docs/benchmark.md](docs/benchmark.md) belgelerinde şeffaf donanım ve istatistik raporu.

### 2. 🛡️ Üretim Sertleştirme & Dinamik CPUID Dispatching
- C++ çekirdeğine `nv::has_avx2_hardware()` kontrolü eklendi. Eski CPU ve ARM (Apple Silicon) cihazlarında `SIGILL` çökme hatası riski sıfırlandı.
- Python runtime teşhis fonksiyonu: `nv._core.has_avx2()`.

### 3. 📦 Kanal Bazlı (Per-Channel) INT8 Kuantizasyon Engine
- `quantize_tensor_per_channel` ile filtre bazlı bağımsız ölçekleme ($S_i$).
- Aşırı uç değerlerin (outliers) hassasiyeti bozması engellendi. **%74.96 bellek küçültme**.
- Deney #01 (Reproducible Seed=42) ile **+%0.00% Accuracy Drop** (doğruluk kaybı) tescillendi.

### 4. 📂 Ayrıştırılmış Örnekler & Öğreticiler (`examples/` vs `tutorials/`)
- **`examples/`:** 5 hızlı kod örneği (`01_lenet`, `02_resnet`, `03_vit`, `04_quantization`, `05_onnx`).
- **`tutorials/`:** 
  - `01_train_mnist.py`: Tekrarlanabilir MNIST Eğitimi & INT8 Accuracy Drop Deneyi.
  - `live_3stage_demo.py`: 3 Aşamalı Canlı Karşılaştırma, Hibrit Kullanım & Saf Eğitim.

### 5. 🔬 Akademik Araştırma Klasörü (`research/`)
- `research/0001-im2col-vs-direct-conv.md`: `im2col + GEMM` Önbellek Verimliliği Analizi (10x Hızlanma).
- `research/0003-int8-quantization.md`: Per-Channel INT8 Matematiği.

### 6. 📜 Açık Kaynak Yönetişim Belgeleri
- `CONTRIBUTING.md`: Katkı Sağlama Rehberi.
- `SECURITY.md`: Güvenlik Politikası ve Gizli Açık Bildirimi.
- `CHANGELOG.md`: Semantik Sürüm Geçmişi.
- `WIKI.md`: GitHub Wiki Rehberi.

---

## 🛠️ Kurulum

```bash
pip install newton-vision==1.1.0
```

---

## 🧪 Doğrulama ve Testler

Tüm birim ve entegrasyon testleri `%100 PASS` ile sonuçlanmıştır.
