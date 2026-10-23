# 📊 newton-vision v1.1.6 — Kapsamlı Bilimsel Benchmark Raporu

> **Felsefe:** Ölçülmemiş iddialar yok; test edilen ortam, girdi boyutu, model derinliği ve ölçüm kapsamı (Salt Çıkarım vs Uçtan Uca) şeffafça ayrıştırılmıştır.  
> **Donanım:** `Intel64 Family 6 / AMD64 Architecture` | Threads: `12 OpenMP Threads` | SIMD: `AVX2 Enabled` | Precision: `FP32 / INT8`  
> **Referans:** `PyTorch v2.6.0+cpu`

---

## 1. KATEGORİ 1: MATHEMATICAL PARITY (MATEMATİKSEL PARİTE)
*Aynı PyTorch ağırlıkları aktarıldıktan sonra, aynı girdi tensörü karşısında üretilen çıktının doğruluğu:*

* **Test Modeli:** 4-Katmanlı CNN (Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear)
* **Girdi Tensörü:** `(1, 3, 32, 32)` FP32 rastgele tensör
* **Maksimum Mutlak Fark (Max Abs Diff):** `0.00000008` (100 milyonda 8)
* **Ortalama Mutlak Fark (Mean Abs Diff):** `0.00000004`
* **Kosinüs Benzerliği (Cosine Similarity):** `%100.000000`
* **Bilimsel Sonuç:** Test edilen model ve giriş üzerinde, newton-vision çıktısı PyTorch referansına **floating-point toleransı (`8e-8`) dahilinde tam sayısal parite** göstermiştir.

---

## 2. KATEGORİ 2: INT8 KUANTİZE PARAMETRE BELLEK TASARRUFU
*Modelin kuantize edilen katman ağırlıklarının bellek ayak izi kıyaslaması:*

| Ölçülen Metrik | FP32 Değeri | INT8 Değeri | Tasarruf |
|---|---|---|---|
| **Kuantize Parametre Sayısı** | 268,464 parametre | 268,464 parametre | — |
| **Kuantize Parametre Belleği** | **1,048.69 KB** | **262.37 KB** | **%74.98 Tasarruf** |

> *Açıklama: Bu tasarruf yalnızca kuantize edilen katmanların tensör ağırlıklarını ifade eder; tüm uygulama çalışma zamanı (runtime) belleğini kapsamaz.*

---

## 3. KATEGORİ 3: KERNEL PERFORMANCE (SALT ÇEKİRDEK HESAPLAMA)
*Disk G/Ç ve görsel ön işleme hariç, doğrudan RAM üzerindeki tensörlerin C++ AVX2 forward pass süresi:*

| Model Mimarisi | Girdi Boyutu | newton-vision Latency | newton-vision FPS | PyTorch CPU Latency | PyTorch CPU FPS | Durum / Karşılaştırma |
|---|---|---|---|---|---|---|
| **LeNet-5 (Pure)** | `(1, 1, 28, 28)` | **0.74 ms** | **1,352 FPS** | — | — | Ultra-Hafif Çıkarım |
| **4-Katmanlı CNN (FP32)** | `(1, 3, 32, 32)` | **1.28 ms** | **779 FPS** | 1.15 ms | 869 FPS | Denk Seviye (%89 Parite) |
| **4-Katmanlı CNN (INT8)** | `(1, 3, 32, 32)` | **1.54 ms** | **649 FPS** | — | — | INT8 per-channel ağırlık (FP32'den yavaş; hesap INT8 SIMD değil) |
| **ViT-Tiny (Transformer)** | `(1, 3, 224, 224)` | **61.25 ms** | **16.3 FPS** | 24.18 ms | 41.3 FPS | 2.53x Yavaş *(aynı boyutlu PyTorch ViT-Tiny)* |
| **ResNet-18 (8 Residual Blok)**| `(1, 3, 224, 224)` | **122.97 ms** | **8.1 FPS** | 29.34 ms | 34.1 FPS | 4.19x Yavaş *(PyTorch oneDNN/MKL avantajı)* |
| **MobileNetV2 (Depthwise)** | `(1, 3, 224, 224)` | **109.18 ms** | **9.2 FPS** | 25.18 ms | 39.7 FPS | 4.34x Yavaş *(C++ optimizasyonundan önce 1,565 ms / ~73x)*|

> 224x224 satırları 2026-10-23'te `python benchmarks/bench_vs_pytorch.py` ile ölçüldü (`eval + no_grad`, `Warmup=3, Runs=10`, parametre sayıları eşit; ham rapor `BENCH_VS_PYTORCH.md`). Mutlak süreler koşudan koşuya değişkendir.
>
> **Düzeltme:** Önceki ViT-Tiny satırı (210.62 ms vs 479.76 ms, "2.28x Daha Hızlı") geçersizdir. Script PyTorch tarafında ViT-B/16 (86M parametre) ölçüyordu.

---

## 4. KATEGORİ 4: END-TO-END APPLICATION PERFORMANCE (UÇTAN UCA BORU HATTI)
*Gerçek dünya senaryosu: Diskten görsel okuma, PIL renk düzeltme, Bounding Box çıkarma, merkezleme ve tahmin:*

* **Uygulama:** `apps/image_predictor.py` (Kullanıcının sağladığı gerçek el yazısı 323x323 JPEG görseli)
* **Adım 1:** Diskten 323x323 JPEG okuma ve gri tonlama dönüşümü
* **Adım 2:** Arka plan eşikleme, rakam sınır kutusunu (bounding box) bulma
* **Adım 3:** 20x20 oran koruyarak ölçekleme ve 28x28 kanvasa ortalama
* **Adım 4:** C++ LeNet-5 tensör çıkarımı + ASCII terminal render
* **Toplam Uçtan Uca Gecikme:** **1.50 – 1.99 ms / dosya (~500 – 660 FPS)**
* **Tahmin Başarısı:** '5' Rakamı (%94.43 Güven Skoru)

---

## 5. BİLİMSEL SONUÇ VE MÜHENDİSLİK ŞEFFAFLIĞI

1. **Neredeyiz:** Küçük girdilerde (4 katmanlı CNN, 32x32) PyTorch CPU ile benzer gecikme ölçülmüştür (1.28 ms vs 1.15 ms). Sayısal parite güçlüdür.
2. **Nerede Arkadayız:** Ölçülen tüm büyük modellerde PyTorch CPU öndedir: ResNet-18 ~4.2x, MobileNetV2 ~4.3x, ViT-Tiny ~2.5–3.7x. Bu sonuç yol haritasındaki optimizasyon hedeflerini belirler.
