# 🏛️ newton-vision Sistem Mimarısı (System Architecture)

`newton-vision`, C++17 OpenMP ve NVIDIA CUDA çekirdeği üzerine inşa edilmiş, hafif ve modüler bir derin öğrenme altyapısıdır.

---

## 📐 Modüler Katman Mimarisi

```
+-------------------------------------------------------------------+
|                        Python Kullanıcı Katmanı                   |
|   nv.Tensor | nv.optics (Conv2d, ViT) | nv.gravity | nv.models    |
+-------------------------------------------------------------------+
                                  |
                                  v  (pybind11 C++ Binding)
+-------------------------------------------------------------------+
|                      C++17 Standalone Engine                      |
|           nv::Tensor | nv::Conv2d | nv::Linear | nv::Model        |
+-------------------------------------------------------------------+
                 |                                  |
                 v                                  v
+----------------------------------+  +-----------------------------+
|    CPU Parallel GEMM & AVX2      |  |     NVIDIA CUDA GPU Core    |
|   (OpenMP + 256-bit SIMD FMA)    |  |  (conv2d, relu, maxpool)    |
+----------------------------------+  +-----------------------------+
```

---

## 🔑 Temel Donanım ve Bellek Prensipleri

1. **Yönlü Türev Grafı (DAG & Autograd):** Tensörler üzerindeki her matematiksel işlem geriye yayılım için graf düğümleri oluşturur (`nv.fluxion`). Graf Python'da tutulur; tensör verisi NumPy `float32` dizileridir.
2. **NumPy View'ları:** `reshape`, `transpose` ve basit dilimleme forward'da NumPy'ın view semantiğini kullanır (mümkünse kopya yapılmaz). Backward'da gradyanlar yeni dizilere yazılır (ör. dilimleme için `np.add.at` ile tam boyutlu sıfır dizisi). C++ çekirdeklerine geçerken pybind11 `forcecast`/`c_style` gerekiyorsa kopya oluşturur.
3. **C++ & Python Çift API:** Modeller Python'da `nv.models` ile çalışır. Ayrıca `include/newton/` altında bir C++ çıkarım API'si (`newton.hpp`) vardır. Header-only değildir, `cpp/layers.cpp` ile birlikte derlenmesi gerekir; Python modelleri `.nv.bin` formatında dışa aktarılıp burada yüklenir.
4. **Cihaz:** CUDA kernel'ları (`conv2d` stride/padding/dilation/groups, `relu`, `maxpool2d`; yalnızca forward) sadece CUDA ile derlenen Python modülünde (`scripts/build_msvc_cuda.bat`) bulunur. Her çağrıda host ↔ device kopyası yapılır ve CUDA hataları `RuntimeError` olarak fırlatılır. Kalıcı GPU belleği yönetimi yoktur.
