# NEWTON-VISION — RESMİ SAYISAL PARİTE VE DOĞRULAMA MATRİSİ RAPORU

> **Rapor Türü:** Bağımsız, Deterministik ve Parametrik Karşılaştırma  
> **Tarih / Zaman:** 2026-10-23 22:39:04  
> **Framework Sürümü:** newton-vision v1.1.6  
> **Referans Motor:** PyTorch 2.6.0+cpu (IEEE 754 CPU Referansı)  
> **Donanım Platformu:** Intel64 Family 6 Model 141 Stepping 1, GenuineIntel (Windows 11)  
> **Deterministik Tohum:** `SEED = 42`  

---

## 1. ÖZET DOĞRULAMA SKORU

* **Kapsam:** 8 Kategori, 24 Parametrik Test Senaryosu, 85 Ayrık Tensör Ölçüm Noktası (Forward, Input Grad, Weight Grad, Bias Grad, Optimizer Moments)
* **Başarılı (Tolerans İçi):** 85 / 85
* **Başarısız (Tolerans Dışı):** 0
* **Parite Başarı Oranı:** **%100.00**

---

## 2. HATA ÖLÇÜM METODOLOJİSİ VE METRİKLER

1. **Maksimum Mutlak Hata (Max Abs Diff):**
   $$\text{MaxAbs} = \max |y_{\text{nv}} - y_{\text{ref}}|$$
2. **Maksimum Göreli Hata (Max Rel Diff):**
   $$\text{MaxRel} = \max \frac{|y_{\text{nv}} - y_{\text{ref}}|}{\max(|y_{\text{nv}}|, |y_{\text{ref}}|) + 10^{-7}}$$
3. **Kosinüs Benzerliği (Cosine Similarity):**
   $$\text{CosSim} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

> **Tolerans Politikası:**
> * **Kesin Autograd & Forward:** $\text{MaxAbs} \le \text{atol} \ (10^{-4})$ veya $\text{MaxRel} \le \text{rtol} \ (10^{-3})$ VE $\text{CosSim} \ge 0.9999$.
> * **Kayıplı INT8 Sıkıştırma:** 8-bit kuantizasyon kayıplı olduğundan ve sıfır civarında bağıl hata yapay büyüdüğünden $\text{CosSim} \ge 0.999$ ve $\text{MaxAbs} \le 1.0$ esas alınır.

---

## 3. PARAMETRİK TEST SONUÇLARI TABLOSU

| Kategori | Test Konfigürasyonu | Ölçülen Bileşen | Tolerans | Max Abs Diff | Mean Abs Diff | Max Rel Diff | Kosinüs Benzerliği | Durum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| Adam Trajectory | 1 Steps (lr=0.01, wd=0.0) | Weights | `atol=1e-04, rtol=1e-03` | `3.73e-09` | `3.73e-11` | `6.48e-08` | `%100.0000` | **PASS** |
| Adam Trajectory | 1 Steps (lr=0.01, wd=0.0) | 1st Moment (m) | `atol=1e-04, rtol=1e-03` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `%100.0000` | **PASS** |
| Adam Trajectory | 1 Steps (lr=0.01, wd=0.0) | 2nd Moment (v) | `atol=1e-04, rtol=1e-03` | `4.66e-10` | `2.10e-11` | `1.14e-07` | `%100.0000` | **PASS** |
| Adam Trajectory | 3 Steps (lr=0.005, wd=0.01) | Weights | `atol=1e-04, rtol=1e-03` | `2.98e-08` | `3.03e-10` | `9.80e-07` | `%100.0000` | **PASS** |
| Adam Trajectory | 3 Steps (lr=0.005, wd=0.01) | 1st Moment (m) | `atol=1e-04, rtol=1e-03` | `2.98e-08` | `5.69e-09` | `1.26e-06` | `%100.0000` | **PASS** |
| Adam Trajectory | 3 Steps (lr=0.005, wd=0.01) | 2nd Moment (v) | `atol=1e-04, rtol=1e-03` | `9.31e-10` | `8.10e-11` | `1.58e-07` | `%100.0000` | **PASS** |
| Adam Trajectory | 10 Steps (lr=0.001, wd=0.05) | Weights | `atol=1e-04, rtol=1e-03` | `7.45e-09` | `1.02e-10` | `2.02e-07` | `%100.0000` | **PASS** |
| Adam Trajectory | 10 Steps (lr=0.001, wd=0.05) | 1st Moment (m) | `atol=1e-04, rtol=1e-03` | `8.94e-08` | `2.06e-08` | `4.59e-06` | `%100.0000` | **PASS** |
| Adam Trajectory | 10 Steps (lr=0.001, wd=0.05) | 2nd Moment (v) | `atol=1e-04, rtol=1e-03` | `1.86e-09` | `1.98e-10` | `1.30e-07` | `%100.0000` | **PASS** |
| Conv2d Matrix | Standard 3x3 | Forward (Y) | `atol=1e-04, rtol=1e-03` | `3.27e-06` | `3.59e-07` | `2.92e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Standard 3x3 | Input Grad | `atol=1e-04, rtol=1e-03` | `3.79e-06` | `4.80e-07` | `7.87e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Standard 3x3 | Weight Grad | `atol=1e-04, rtol=1e-03` | `1.79e-05` | `3.39e-06` | `1.14e-04` | `%100.0000` | **PASS** |
| Conv2d Matrix | Standard 3x3 | Bias Grad | `atol=1e-04, rtol=1e-03` | `4.44e-06` | `1.86e-06` | `8.97e-07` | `%100.0000` | **PASS** |
| Conv2d Matrix | Strided 3x3 s=2 | Forward (Y) | `atol=1e-04, rtol=1e-03` | `3.96e-06` | `4.41e-07` | `2.41e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Strided 3x3 s=2 | Input Grad | `atol=1e-04, rtol=1e-03` | `2.20e-06` | `1.78e-07` | `2.55e-03` | `%100.0000` | **PASS** |
| Conv2d Matrix | Strided 3x3 s=2 | Weight Grad | `atol=1e-04, rtol=1e-03` | `6.71e-06` | `1.18e-06` | `6.94e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Strided 3x3 s=2 | Bias Grad | `atol=1e-04, rtol=1e-03` | `4.44e-06` | `1.80e-06` | `1.86e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated 3x3 d=2 | Forward (Y) | `atol=1e-04, rtol=1e-03` | `3.10e-06` | `3.75e-07` | `2.20e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated 3x3 d=2 | Input Grad | `atol=1e-04, rtol=1e-03` | `2.10e-06` | `2.79e-07` | `3.14e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated 3x3 d=2 | Weight Grad | `atol=1e-04, rtol=1e-03` | `2.73e-05` | `4.12e-06` | `1.25e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated 3x3 d=2 | Bias Grad | `atol=1e-04, rtol=1e-03` | `7.24e-06` | `4.49e-06` | `4.11e-07` | `%100.0000` | **PASS** |
| Conv2d Matrix | Grouped G=2 | Forward (Y) | `atol=1e-04, rtol=1e-03` | `2.94e-06` | `3.00e-07` | `2.37e-04` | `%100.0000` | **PASS** |
| Conv2d Matrix | Grouped G=2 | Input Grad | `atol=1e-04, rtol=1e-03` | `1.50e-06` | `2.30e-07` | `5.34e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Grouped G=2 | Weight Grad | `atol=1e-04, rtol=1e-03` | `1.21e-05` | `2.48e-06` | `8.21e-04` | `%100.0000` | **PASS** |
| Conv2d Matrix | Grouped G=2 | Bias Grad | `atol=1e-04, rtol=1e-03` | `3.10e-06` | `1.99e-06` | `1.62e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated+Grouped | Forward (Y) | `atol=1e-04, rtol=1e-03` | `2.27e-06` | `1.93e-07` | `3.97e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated+Grouped | Input Grad | `atol=1e-04, rtol=1e-03` | `1.97e-06` | `1.70e-07` | `1.27e-04` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated+Grouped | Weight Grad | `atol=1e-04, rtol=1e-03` | `1.17e-05` | `3.25e-06` | `5.13e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Dilated+Grouped | Bias Grad | `atol=1e-04, rtol=1e-03` | `6.85e-06` | `4.74e-06` | `1.58e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Depthwise G=Cin | Forward (Y) | `atol=1e-04, rtol=1e-03` | `9.22e-07` | `1.13e-07` | `1.64e-03` | `%100.0000` | **PASS** |
| Conv2d Matrix | Depthwise G=Cin | Input Grad | `atol=1e-04, rtol=1e-03` | `9.89e-07` | `1.06e-07` | `6.26e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Depthwise G=Cin | Weight Grad | `atol=1e-04, rtol=1e-03` | `9.14e-06` | `1.65e-06` | `2.54e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Depthwise G=Cin | Bias Grad | `atol=1e-04, rtol=1e-03` | `4.03e-06` | `2.42e-06` | `1.31e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Kernel 1x1 s=1 | Forward (Y) | `atol=1e-04, rtol=1e-03` | `8.86e-07` | `1.15e-07` | `2.04e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Kernel 1x1 s=1 | Input Grad | `atol=1e-04, rtol=1e-03` | `1.85e-06` | `2.23e-07` | `4.54e-05` | `%100.0000` | **PASS** |
| Conv2d Matrix | Kernel 1x1 s=1 | Weight Grad | `atol=1e-04, rtol=1e-03` | `4.98e-06` | `1.22e-06` | `2.34e-06` | `%100.0000` | **PASS** |
| Conv2d Matrix | Kernel 1x1 s=1 | Bias Grad | `atol=1e-04, rtol=1e-03` | `3.45e-06` | `1.28e-06` | `4.96e-06` | `%100.0000` | **PASS** |
| Linear Matrix | 2D Dense (N=8, 32->16) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `1.80e-06` | `3.74e-07` | `1.30e-05` | `%100.0000` | **PASS** |
| Linear Matrix | 2D Dense (N=8, 32->16) | Input Grad | `atol=1e-04, rtol=1e-03` | `1.29e-06` | `1.80e-07` | `2.34e-05` | `%100.0000` | **PASS** |
| Linear Matrix | 2D Dense (N=8, 32->16) | Weight Grad | `atol=1e-04, rtol=1e-03` | `6.26e-07` | `9.63e-08` | `8.10e-06` | `%100.0000` | **PASS** |
| Linear Matrix | 2D Dense (N=8, 32->16) | Bias Grad | `atol=1e-04, rtol=1e-03` | `2.98e-07` | `1.46e-07` | `9.63e-07` | `%100.0000` | **PASS** |
| Linear Matrix | 3D Batch-Seq (2, 10, 64->32) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `9.19e-06` | `8.14e-07` | `4.65e-05` | `%100.0000` | **PASS** |
| Linear Matrix | 3D Batch-Seq (2, 10, 64->32) | Input Grad | `atol=1e-04, rtol=1e-03` | `2.18e-06` | `2.68e-07` | `3.15e-05` | `%100.0000` | **PASS** |
| Linear Matrix | 3D Batch-Seq (2, 10, 64->32) | Weight Grad | `atol=1e-04, rtol=1e-03` | `2.08e-06` | `2.46e-07` | `1.93e-03` | `%100.0000` | **PASS** |
| Linear Matrix | 3D Batch-Seq (2, 10, 64->32) | Bias Grad | `atol=1e-04, rtol=1e-03` | `1.68e-06` | `3.85e-07` | `5.02e-07` | `%100.0000` | **PASS** |
| Linear Matrix | 3D ViT Patch (1, 16, 128->64) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `1.23e-05` | `1.72e-06` | `1.41e-04` | `%100.0000` | **PASS** |
| Linear Matrix | 3D ViT Patch (1, 16, 128->64) | Input Grad | `atol=1e-04, rtol=1e-03` | `6.92e-06` | `8.23e-07` | `3.38e-05` | `%100.0000` | **PASS** |
| Linear Matrix | 3D ViT Patch (1, 16, 128->64) | Weight Grad | `atol=1e-04, rtol=1e-03` | `1.97e-06` | `2.05e-07` | `1.15e-03` | `%100.0000` | **PASS** |
| Linear Matrix | 3D ViT Patch (1, 16, 128->64) | Bias Grad | `atol=1e-04, rtol=1e-03` | `9.83e-07` | `2.33e-07` | `8.41e-07` | `%100.0000` | **PASS** |
| BatchNorm2d | Train Mode (N=4, C=3, 6x6) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `3.85e-07` | `5.67e-08` | `9.27e-05` | `%100.0000` | **PASS** |
| BatchNorm2d | Train Mode | Input Grad | `atol=1e-04, rtol=1e-03` | `7.17e-07` | `6.50e-08` | `8.51e-06` | `%100.0000` | **PASS** |
| BatchNorm2d | Train Mode | Gamma Grad | `atol=1e-04, rtol=1e-03` | `7.54e-07` | `4.92e-07` | `6.44e-08` | `%100.0000` | **PASS** |
| BatchNorm2d | Train Mode | Beta Grad | `atol=1e-04, rtol=1e-03` | `1.60e-06` | `7.17e-07` | `5.48e-07` | `%100.0000` | **PASS** |
| LayerNorm | 3D Shape (2, 8, 32) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `6.20e-07` | `5.79e-08` | `1.85e-05` | `%100.0000` | **PASS** |
| LayerNorm | 3D Shape | Input Grad | `atol=1e-04, rtol=1e-03` | `7.34e-07` | `5.77e-08` | `5.64e-06` | `%100.0000` | **PASS** |
| LayerNorm | 3D Shape | Gamma Grad | `atol=1e-04, rtol=1e-03` | `1.23e-06` | `2.65e-07` | `3.18e-06` | `%100.0000` | **PASS** |
| LayerNorm | 3D Shape | Beta Grad | `atol=1e-04, rtol=1e-03` | `6.46e-07` | `2.62e-07` | `2.75e-06` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `6.86e-07` | `1.34e-07` | `6.55e-06` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Input Grad | `atol=1e-04, rtol=1e-03` | `1.18e-06` | `2.43e-07` | `9.55e-05` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Q Proj Grad | `atol=1e-04, rtol=1e-03` | `1.72e-06` | `3.69e-07` | `1.45e-05` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | K Proj Grad | `atol=1e-04, rtol=1e-03` | `1.51e-06` | `3.86e-07` | `1.47e-04` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | V Proj Grad | `atol=1e-04, rtol=1e-03` | `2.27e-06` | `2.91e-07` | `9.33e-05` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Out Proj Grad | `atol=1e-04, rtol=1e-03` | `9.69e-07` | `2.20e-07` | `1.95e-05` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Q Bias Grad | `atol=1e-04, rtol=1e-03` | `1.25e-06` | `4.45e-07` | `2.56e-06` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Tiny (B=2, S=6, D=16, H=2) | Out Bias Grad | `atol=1e-04, rtol=1e-03` | `6.59e-07` | `1.31e-07` | `6.47e-07` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `1.32e-06` | `2.64e-07` | `2.48e-05` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Input Grad | `atol=1e-04, rtol=1e-03` | `2.17e-06` | `4.78e-07` | `4.77e-04` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Q Proj Grad | `atol=1e-04, rtol=1e-03` | `4.63e-06` | `6.62e-07` | `3.65e-04` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | K Proj Grad | `atol=1e-04, rtol=1e-03` | `5.21e-06` | `6.63e-07` | `1.09e-03` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | V Proj Grad | `atol=1e-04, rtol=1e-03` | `2.55e-06` | `4.69e-07` | `1.53e-04` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Out Proj Grad | `atol=1e-04, rtol=1e-03` | `3.99e-06` | `5.38e-07` | `3.31e-04` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Q Bias Grad | `atol=1e-04, rtol=1e-03` | `1.76e-06` | `7.85e-07` | `1.34e-06` | `%100.0000` | **PASS** |
| MHSA Matrix | MHSA Small (B=1, S=12, D=32, H=4) | Out Bias Grad | `atol=1e-04, rtol=1e-03` | `4.17e-07` | `1.31e-07` | `3.02e-06` | `%100.0000` | **PASS** |
| CrossEntropy | Hard Targets (N=8, C=5) | Loss Value | `atol=1e-04, rtol=1e-03` | `1.77e-07` | `1.77e-07` | `7.77e-08` | `%100.0000` | **PASS** |
| CrossEntropy | Hard Targets | Logits Grad | `atol=1e-04, rtol=1e-03` | `3.57e-09` | `1.24e-09` | `1.76e-07` | `%100.0000` | **PASS** |
| CrossEntropy | Soft Targets (Sum=1.0) | Loss Value | `atol=1e-04, rtol=1e-03` | `4.84e-08` | `4.84e-08` | `2.98e-08` | `%100.0000` | **PASS** |
| CrossEntropy | Soft Targets | Logits Grad | `atol=1e-04, rtol=1e-03` | `2.32e-08` | `5.55e-09` | `3.97e-07` | `%100.0000` | **PASS** |
| CrossEntropy | Extreme (+-80.0) | Loss Value | `atol=1e-04, rtol=1e-03` | `6.87e-10` | `6.87e-10` | `6.82e-03` | `%0.0000` | **PASS** |
| Activations | GELU (approx='tanh') | Forward (Y) | `atol=1e-04, rtol=1e-03` | `3.79e-07` | `5.07e-08` | `1.73e-01` | `%100.0000` | **PASS** |
| Activations | GELU (approx='tanh') | Input Grad | `atol=1e-04, rtol=1e-03` | `9.08e-07` | `1.28e-07` | `2.67e-01` | `%100.0000` | **PASS** |
| Activations | Sigmoid Extreme (+-500) | Forward (Y) | `atol=1e-04, rtol=1e-03` | `1.89e-08` | `4.26e-09` | `4.04e-08` | `%100.0000` | **PASS** |
| Activations | Sigmoid Extreme (+-500) | Input Grad | `atol=1e-04, rtol=1e-03` | `7.62e-09` | `2.13e-09` | `3.88e-08` | `%100.0000` | **PASS** |
| INT8 Quantization | Symmetric [-127, 127] Range | Bounds Check | `cos>=0.999` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `%100.0000` | **PASS** |
| INT8 Quantization | Reconstruction Error | FP32 Recovery | `cos>=0.999` | `3.57e-02` | `1.79e-02` | `1.00e+00` | `%99.9964` | **PASS** |
| INT8 Quantization | QuantizedLinear vs FP32 Dense | Forward (Y) | `cos>=0.999` | `5.07e-01` | `1.64e-01` | `3.82e-01` | `%99.9946` | **PASS** |

---

## 4. KAPSAM VE GEÇERLİLİK BEYANI

1. **Tekrarlanabilirlik:** Aynı yazılım sürümleri ve eşdeğer yürütüm koşulları altında testler deterministik seed (`SEED=42`) ile tekrarlanabilir olacak şekilde tasarlanmıştır; farklı CPU mimarileri, derleyiciler veya OpenMP iş parçacığı toplama sıralarından kaynaklanabilecek kayan nokta son-bit farkları için tanımlı sayısal toleranslar (`atol`, `rtol`) esas alınır.
2. **Terminoloji Ayrımı:** `groups=Cin=Cout` konfigürasyonu salt **Depthwise Convolution** işlemidir. Endüstri standardı Depthwise Separable Convolution mimarisi, bu katmanın ardına $1\times 1$ pointwise evrişim eklenmesiyle elde edilir.
3. **Kapsam Sınırı:** Bu rapor, test edilen 24 parametrik senaryo ve 85 tensör çıktısı dahilinde newton-vision motorunun PyTorch referansı ile floating-point toleransı dahilinde parite sağladığını doğrular; henüz test edilmemiş olası sonsuz girdi uzayını veya dinamik bellek sınırlarını tek başına kapsamaz.

*Rapor `benchmarks/verify_parity_matrix.py` tarafından otomatik üretilmiştir.*
