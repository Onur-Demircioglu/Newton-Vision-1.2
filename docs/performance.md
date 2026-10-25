# Performans ve Benchmark

Tüm ölçümler bu makinede (Intel64 Family 6, 12 iş parçacığı, GTX 1650) yapılmıştır; mutlak süreler koşudan koşuya değişir. Ham çıktılar: [`benchmarks/BENCH_VS_PYTORCH.md`](../benchmarks/BENCH_VS_PYTORCH.md), [`benchmarks/VERIFICATION_REPORT.md`](../benchmarks/VERIFICATION_REPORT.md). Hepsini yeniden çalıştırmak için: `python benchmarks/run_all.py --full`.

## 📊 Benchmark Sonuçları & Performans

### 1. Sistem Çıkarım Performansı: Salt Çekirdek vs. Uçtan Uca Boru Hattı

Sistem performansını değerlendirirken disk I/O ve görsel önişleme maliyetlerini matris motoru maliyetinden **kesin olarak ayırıyoruz** (`Intel Core CPU / AVX2, LeNet-5 28x28 Grayscale`):

| Aşama | Süre (Gecikme) | Hız (FPS) | Kapsam |
| :--- | :--- | :--- | :--- |
| **Salt Tensör Çıkarımı (Pure Forward)** | **1.163 ms** | **859.6 FPS** | Yalnızca C++ AVX2 matris hesaplaması ve aktivasyonlar |
| **Uçtan Uca Boru Hattı (Full E2E)** | **4.020 ms** | **248.8 FPS** | Disk I/O + PIL Görsel Okuma + Normalizasyon + Çıkarım |

> **Mühendislik Notu:** Saf matris çıkarımı ile disk okuma içeren uçtan uca süre arasındaki fark I/O ve Python nesne serileştirme maliyetidir. İki metrik birbiriyle karıştırılmamalıdır.

---

### 2. INT8 Parametre Kuantizasyonu ve Bellek Tasarrufu

Ağırlıklar için simetrik $[-127, 127]$ aralığı ($Z=0$) ve filtre başına per-channel ölçekleme kullanılır:

* **FP32 Parametre Bellek Boyutu:** `1048.69 KB`
* **INT8 Parametre Bellek Boyutu:** `262.37 KB`
* **Bellek Tasarrufu:** **%74.98** *(Teorik %75 sınırına tam parite)*
* **Kuantize Model Çıktı Rekonstrüksiyon Kosinüs Benzerliği:** **%99.9964**

---

### 3. Model Zoo Karşılaştırması (PyTorch CPU vs newton-vision)

`Intel64 Family 6 (12 Threads)` üzerinde PyTorch v2.6.0 CPU ile karşılaştırma. Ölçüm: `python benchmarks/bench_vs_pytorch.py`, 2026-10-23, `Warmup=3, Runs=10`, `(1, 3, 224, 224)`, iki tarafta da `eval + no_grad`. Script parametre sayıları eşleşmezse durur. Ham rapor: [`benchmarks/BENCH_VS_PYTORCH.md`](../benchmarks/BENCH_VS_PYTORCH.md).

| Model | Framework | Parametre | Mean (ms) | Std (ms) | FPS | Karşılaştırma |
|---|---|---|---|---|---|---|
| **ResNet-18** | newton-vision | 11,689,512 | **122.97** | ±6.09 | **8.1** | 4.19x Yavaş |
| | PyTorch CPU | 11,689,512 | 29.34 | ±1.39 | 34.1 | Referans |
| **MobileNetV2** | newton-vision | 3,504,872 | **109.18** | ±12.57 | **9.2** | 4.34x Yavaş |
| | PyTorch CPU | 3,504,872 | 25.18 | ±6.76 | 39.7 | Referans |
| **ViT-Tiny** | newton-vision | 5,717,416 | **61.25** | ±7.03 | **16.3** | 2.53x Yavaş |
| | PyTorch CPU (torchvision `VisionTransformer`, aynı boyutlar) | 5,717,416 | 24.18 | ±1.21 | 41.3 | Referans |

> **Ölçüm değişkenliği:** Bu dizüstü makinede tekrarlı koşularda mutlak süreler belirgin şekilde değişiyor (ör. ViT-Tiny 58–84 ms; ViT kodu değişmediği halde oranı 2.5–3.7x arasında oynadı).
>
> **C++ optimizasyonunun etkisi (aynı gün, önce → sonra):** MobileNetV2 çıkarımı 1565 ms → 109 ms (PyTorch'a göre ~73x → ~4.3x yavaş). Gruplu/depthwise evrişim artık gruplar üzerinden paralel çalışıyor. ResNet-18 ve ViT'nin forward yolu değişmedi.

**Yorum:** Tüm model ölçümlerinde PyTorch CPU öndedir. Fark özellikle evrişim yoğun mimarilerde büyüktür; PyTorch'un oneDNN/MKL backend'i yıllardır optimize edilmektedir, depthwise evrişim için özel bir çekirdek hâlâ yoktur (genel gruplu im2col yolu, gruplar üzerinden paralel). newton-vision'ın değeri hızdan çok iç işleyişin şeffaf ve doğrulanabilir olmasıdır.

---

## Küçük Modeller: Salt Çekirdek Süresi
*Disk G/Ç ve görsel ön işleme hariç, doğrudan RAM üzerindeki tensörlerin C++ AVX2 forward pass süresi:*

| Model Mimarisi | Girdi Boyutu | newton-vision Latency | newton-vision FPS | PyTorch CPU Latency | PyTorch CPU FPS | Durum / Karşılaştırma |
|---|---|---|---|---|---|---|
| **LeNet-5 (Pure)** | `(1, 1, 28, 28)` | **0.74 ms** | **1,352 FPS** | — | — | Ultra-Hafif Çıkarım |
| **4-Katmanlı CNN (FP32)** | `(1, 3, 32, 32)` | **1.28 ms** | **779 FPS** | 1.15 ms | 869 FPS | Denk Seviye (%89 Parite) |
| **4-Katmanlı CNN (INT8)** | `(1, 3, 32, 32)` | **1.54 ms** | **649 FPS** | — | — | INT8 per-channel ağırlık (FP32'den yavaş; hesap INT8 SIMD değil) |

---

`newton-vision`, x86-64 işlemcilerde C++ GEMM çekirdeğinde (evrişimin im2col sonrası matris çarpımı) **256-bit AVX2 FMA (Fused Multiply-Add)** vektör komutlarını kullanır. Bir AVX2 yazmacı 8 adet `float32` taşıdığı için teorik üst sınır skaler koda göre 8x'tir; skaler yola göre gerçek hızlanma bu repoda ayrıca ölçülmemiştir. Linear ve attention katmanlarındaki matris çarpımları bu çekirdeği değil, NumPy'ın BLAS'ını kullanır.

---

## 🔬 Donanım Vektörizasyonu (`cpp/simd_gemm.h`)

İşlemcinin 256-bit vektör yazmaçlarında tek döngüde 8 adet `float32` sayısı aynı anda işlenir:

```cpp
// 8 adet float32 elemanını tek döngüde çarpıp toplama (FMA)
__m256 a_vec = _mm256_set1_ps(A[m * K + k]);
__m256 b_vec = _mm256_loadu_ps(&B[k * N + n]);
c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec); // C = A * B + C
```

---

## 🔁 Evrişim Backward ve Gruplu Evrişim Paralelliği

Profil (2026-10-23): ResNet-18 / MobileNetV2 eğitim adımında sürenin ~%92'si C++ `conv2d` ve `conv2d_backward` içinde geçiyordu; Python autograd'ın payı ihmal edilebilir düzeydeydi. Yapılan iyileştirmeler:

- **`conv2d_backward`:**
  - dW = dY · colᵀ: (çıkış kanalı, k) çiftleri üzerinden paralel, `omp simd` ile vektörize nokta çarpımı.
  - dcol = Wᵀ · dY: satır bazında paralel, l ekseninde ardışık bellek erişimi.
  - `col2im`: girdi kanalı bazında paralel; her kanal yalnızca kendi düzlemine yazdığı için yarış yok.
- **Gruplu / depthwise evrişim** (grup sayısı ≥ iş parçacığı sayısı): Paralellik grup başına küçük paralel bölgeler yerine gruplar üzerinden yapılır ve her iş parçacığı kendi tamponunu kullanır. `im2col` ve GEMM'deki iç paralel bölgeler `if(!omp_in_parallel())` ile iç içe açılmaz.

| Eğitim adımı (batch 2, 224², 12 iş parçacığı) | Önce | Sonra | Hızlanma |
|---|---|---|---|
| ResNet-18 | 7.36 sn | 1.16 sn | 6.3x |
| MobileNetV2 | 5.26 sn | 1.03 sn | 5.1x |

Doğruluk korundu: PyTorch parite matrisi 85/85 (gruplu, dilated ve depthwise evrişim gradyanları dahil). Katman düzeyinde float64 referansına göre hata ~1e-7.

---

## 🛡️ Dynamic CPUID Dispatching

İşlemci mimarisinden bağımsız güvenli çalışma için, derleme zamanındaki statik `-mavx2` zorunluluğu kaldırılmış ve runtime `nv::has_avx2_hardware()` kontrolü eklenmiştir. Eski işlemcilerde veya ARM çiplerinde (Apple M1/M2/M3) kod `SIGILL` hatası vermeden güvenli OpenMP moduna geçer.
