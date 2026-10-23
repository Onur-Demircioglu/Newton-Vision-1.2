# 🌌 newton-vision Yetenekler ve Eğitim Kapasitesi Kılavuzu

Bu kılavuz, `newton-vision` (v1.1.6) framework'ü ile **neyin eğitilebileceğini**, **donanımsal/yazılımsal sınırları** ve projenin **gelecek yol haritasını** açıklar.

---

## 🟢 1. En Fazla Neler Eğitebiliriz? (Aktif Yetenekler)

Sahip olduğumuz Autograd operasyonları (`@`, `transpose`, `reshape`, `softmax`, `mean`, `sum`, `tile`, `concatenate`) ve görsel katmanlar (`Conv2d`, `BatchNorm2d`, `LayerNorm`, `MultiHeadAttention`, `GELU`) ile şu görevleri sıfırdan eğitebiliriz:

> **Kanıt düzeyi:** Aşağıda "doğrulandı" olarak işaretlenenler repoda çalışan bir script ile gösterilmiştir. "Mümkün" olanlar için gerekli yapı taşları mevcuttur, ancak repoda uçtan uca bir eğitim örneği yoktur.

### A. Görsel Sınıflandırma (Image Classification)
* **MNIST (doğrulandı, 2026-10-23):** Tam MNIST'te LeNet 3 epoch'ta %98.43 test doğruluğu (~80 sn/epoch). Hızlı örnek: `tutorials/01_train_mnist.py` (6000 görsel, 2 epoch, ~%90). Ayrıntılı metrikler: `tutorials/train_and_eval_lenet.py`.
* **Fashion-MNIST / CIFAR-10 (mümkün):** Aynı katmanlarla küçük CNN'ler kurulabilir; repoda bu veri setleri için eğitim örneği yoktur.
* **Özel Sınıflandırma (mümkün):** ViT-Tiny ile kendi veri kümenizi eğitmek mümkündür. CPU'da 224x224 eğitim yavaştır: batch 2 ile bir eğitim adımı ResNet-18 / MobileNetV2'de ~1 sn sürer (süre C++ evrişim çekirdeklerinde geçer).

### B. Dizi Modelleri (mümkün; yalnızca sentetik test örneği var)
* `MultiHeadAttention` + `LayerNorm` + `GELU` ile **sürekli değerli** diziler (ör. sensör / zaman serisi verisi) üzerinde encoder modelleri kurulabilir.
* **Dizi / metin sınıflandırma (mümkün):** `Embedding` + `Conv1d` (veya `MultiHeadAttention`) ile token dizileri sınıflandırılabilir; `tests/test_training_components.py` içinde sentetik bir dizi görevi görülmemiş veride %95'in üzerine çıkıyor. Metni token indekslerine çevirmek için basit kelime/karakter tokenizer'ı `nv.text.Tokenizer` var; alt-kelime (BPE) yöntemi yok.

---

## 🔴 2. Neleri Eğitemeyiz? (Sınırlar ve Kısıtlamalar)

Mühendislik dürüstlüğüyle, mevcut versiyonun mimari sınırları şunlardır:

* **Büyük Dil Modelleri (LLM - GPT / LLaMA):** 
  * *Sebep:* Decoder tabanlı dil modelleri için şart olan **Causal Masked Self-Attention** (yani modelin gelecekteki kelimeleri görmesini engelleyen maskeleme) ve metin tokenleştirici (Tokenizer) altyapısı henüz mevcut değil.
* **Büyük Çözünürlüklü Sıfırdan ImageNet Eğitimi (224x224):**
  * *Sebep:* Dağıtık çoklu GPU eğitimi (Multi-GPU DDP), hafıza tasarruflu attention (FlashAttention) ve FP16/BF16 karma hassasiyet (Mixed Precision) desteği olmadığı için sıfırdan ImageNet eğitimi donanımsal olarak çok uzun sürer. (Bunun yerine PyTorch'tan ağırlık aktarıp çıkarım yapmak mantıklıdır).

---

## 🗺️ 3. Gelecek Yol Haritası (Roadmap)

`newton-vision` projesini daha ileriye taşımak için eklenebilecek olası modüller:

1. **Causal Attention Mask:** `MultiHeadAttention` içerisine maskeleme desteği eklenerek decoder tabanlı üretken (generative) dil modellerinin önünü açmak.
2. **Alt-kelime Tokenizer (BPE/WordPiece):** Kelime/karakter tokenizer'ı (`nv.text.Tokenizer`) mevcut; açık sözlük için alt-kelime yöntemi eklemek.
3. **Mixed Precision (FP16):** Bellek bant genişliğini korumak ve GPU eğitimini 2 kat hızlandırmak için yarı hassasiyetli gradyan hesaplama.
4. **Dağıtık Veri Paralelliği (DDP):** Birden fazla GPU kartının koordineli çalışabilmesini sağlayan iletişim katmanı.

---
> **Not:** newton-vision, derin öğrenmenin ve donanım ivmelendirmesinin mutfağını anlamak için tasarlanmış şeffaf bir eğitim/araştırma framework'üdür.
