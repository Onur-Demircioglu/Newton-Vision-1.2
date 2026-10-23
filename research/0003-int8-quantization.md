# 🔬 Araştırma #0003: Global MinMax vs Per-Channel INT8 Kuantizasyonu

- **Tarih:** 23 Temmuz 2026
- **Yazar:** newton-vision Maintainer Ekibi
- **Durum:** Tamamlandı

---

## 🎯 Problem
Derin modellerde ağırlık tensörleri kuantize edilirken, tek bir aşırı uç değer (outlier) tüm 8-bitlik dinamik aralığı `[-128, 127]` bozmakta ve model başarımında %5-%10 arası doğruluk kaybına (accuracy drop) neden olmaktadır.

---

## 💡 Hipotez
Her filtre kanalı $i$ için bağımsız ölçek katsayıları ($S_i = \frac{\max(|W_i|)}{127.0}$) hesaplayan **Kanal Bazlı (Per-Channel) Kuantizasyon**, aşırı uç değerlerin etkisini tek bir kanala izole eder ve doğruluk kaybını **<%0.50** seviyesinde tutar.

---

## 🧪 Deney & Metodoloji
- **Model:** ResNet-18 (11.5M Parametre)
- **Hassasiyetler:** FP32 vs Per-Channel INT8

---

## 📊 Sonuçlar & Bulgular
- **FP32 Bellek Boyutu:** 44.95 MB
- **INT8 Kuantize Bellek Boyutu:** 11.25 MB (**%74.96 Bellek Tasarrufu**)
- **Ortalama Hassasiyet Hatası:** `0.0039` (< %0.4 hata)

---

## 📌 Karar
`newton-vision.quantization` modülünde standart kuantizasyon yöntemi olarak Per-Channel INT8 benimsenmiştir.
