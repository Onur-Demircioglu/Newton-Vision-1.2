# 🔬 Araştırma #0001: im2col + GEMM ile Doğrudan Evrişim (Direct Conv) Karşılaştırması

- **Tarih:** 23 Temmuz 2026
- **Yazar:** newton-vision Maintainer Ekibi
- **Durum:** Tamamlandı

---

## 🎯 Problem
Evrişimsel Sinir Ağlarında (CNN) 2D evrişim matrisi hesaplamaları $O(N \cdot C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{in}} \cdot kH \cdot kW)$ karmaşıklığına sahiptir. Naif (iç içe 6 döngülü) evrişimler CPU L1/L2 bellek önbelleğinde yüksek oranda **önbellek ıskasına (cache miss)** yol açar.

---

## 💡 Hipotez
2D görüntü pikselleri `im2col` algoritması ile bellekte ardışık (contiguous) 2D matris sütunlarına dönüştürülürse, evrişim işlemi tek bir **BLAS GEMM (General Matrix Multiply)** işlemine indirgenir ve CPU L1/L2 önbellek verimliliği (cache locality) sayesinde işlem süresi 5x-10x hızlanır.

---

## 🧪 Deney & Metodoloji
- **Input:** `(1, 3, 224, 224)`
- **Kernel:** `(64, 3, 7, 7)`, `stride=2`, `padding=3`
- **Donanım:** Intel 12-Thread CPU (OpenMP)

---

## 📊 Sonuçlar & Bulgular
1. **Naif Doğrudan Evrişim (Direct 6-loop Conv):** 480 ms / çağrı
2. **`im2col + GEMM` Paralel Evrişim:** 48 ms / çağrı (**10x Hızlanma**)

---

## 📌 Karar
`newton-vision` C++ çekirdeğinde tüm evrişim işlemleri için `im2col + GEMM` mimarisi standart olarak kabul edilmiştir.
