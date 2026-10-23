# ⚡ AVX2 SIMD Vektörizasyon & Donanım Performansı

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
