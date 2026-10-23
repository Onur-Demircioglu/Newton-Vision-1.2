# 📦 INT8 Kuantizasyon Matematiği & Model Küçültme

`newton_vision.quantization`, 32-bit kayan noktalı (FP32) model ağırlıklarını simetrik 8-bit tam sayı aralığına `[-127, 127]` dönüştürerek **ağırlık belleğini ~%75 küçültür** (ölçek değerleri hariç).

> **Hız notu:** Bu bir bellek optimizasyonudur. `QuantizedConv2d` her forward'da ağırlıkları FP32'ye geri çevirip FP32 evrişim çalıştırır; `QuantizedLinear` INT32 birikimli matris çarpımını NumPy ile yapar. C++ `gemm_int8_cpu` henüz Python'a bağlı değildir. Ölçümlerde INT8 model FP32'den biraz yavaştır.

---

## 📐 Kanal Bazlı (Per-Channel) Ölçekleme Matematiği

Aşırı uç değerlerin (outliers) 8-bitlik dinamik aralığı bozmasını engellemek için her filtre kanalı $i$ için bağımsız ölçek $S_i$ hesaplanır:

$$S_i = \frac{\max(|W_i|)}{127.0}$$

$$W_{i, \text{int8}} = \text{clip}\left(\text{round}\left(\frac{W_i}{S_i}\right), -127, 127\right)$$

(`round` = NumPy `np.round`, ties-to-even.)

---

## 📊 Geri Dönüştürme (Dequantization)

$$W_{i, \text{fp32}} = W_{i, \text{int8}} \cdot S_i$$

Ölçek $S_i = \max(|W_i|)/127$ olduğundan eleman başına yuvarlama hatası en fazla $S_i/2$, yani kanalın maksimum mutlak değerinin **≈ %0.39**'u kadardır ($1/254$).
