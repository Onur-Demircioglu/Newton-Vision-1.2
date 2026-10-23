# 📖 newton-vision Kullanım Kılavuzu & Geliştirici El Kitabı

Bu doküman, **newton-vision** derin öğrenme framework'ünün mimarisini, tüm katmanlarını, eğitim döngüsünü, PyTorch entegrasyonunu ve pratik kullanım senaryolarını adım adım açıklayan resmi kullanım kılavuzudur.

---

## 📑 İçindekiler
1. [Temel Felsefe ve Mimari](#1-temel-felsefe-ve-mimari)
2. [Hızlı Başlangıç (Minimal Örnek)](#2-hızlı-başlangıç-minimal-örnek)
3. [Tensör & Otomatik Türev (Autograd Engine)](#3-tensör--otomatik-türev-autograd-engine)
4. [Katmanlar (Layers API)](#4-katmanlar-layers-api)
5. [Model Konteynırları (Sequential & ModuleList)](#5-model-konteynırları-sequential--modulelist)
6. [Veri Yükleyici (Dataset & DataLoader)](#6-veri-yükleyici-dataset--dataloader)
7. [Kayıp Fonksiyonları & Optimizatörler (Gravity Engine)](#7-kayıp-fonksiyonları--optimizatörler-gravity-engine)
8. [Hazır Modeller (Model Zoo)](#8-hazır-modeller-model-zoo)
9. [PyTorch Ağırlık Aktarımı (State Dict)](#9-pytorch-ağırlık-aktarımı-state-dict)
10. [INT8 Kuantizasyon & Hızlandırma](#10-int8-kuantizasyon--hızlandırma)
11. [Model Kaydetme & Yükleme (JIT / Serialization)](#11-model-kaydetme--yükleme-jit--serialization)
12. [Somut Uygulamalar ve CLI Araçları](#12-somut-uygulamalar-ve-cli-araçları)

---

## 1. Temel Felsefe ve Mimari

`newton-vision`, Python'ın esnekliği ile C++17'nin saf donanım performansını (AVX2 SIMD) birleştiren bağımsız bir derin öğrenme motorudur. 

* **C++ Çekirdeği (`_core.pyd`):** `im2col`, GEMM matris çarpımı, 2D evrişim, havuzlama ve aktivasyon türevleri C++ tarafında çok çekirdekli ve SIMD destekli koşar.
* **Python API (`newton_vision`):** PyTorch `nn.Module` standardında parametre yönetimi, autograd topolojik sıralaması ve veri akışı sağlar.

```
[ Girdi Tensörü ] 
       │
       ▼
[ newton-vision Python API ] (Module, Sequential, Autograd Graph)
       │
       ▼
[ C++17 / AVX2 SIMD Çekirdeği ] (_core: GEMM, Conv2d, im2col, BatchNorm)
       │
       ▼
[ CPU / CUDA Donanım Çıktısı ]
```

---

## 2. Hızlı Başlangıç (Minimal Örnek)

Sıfırdan 3 katmanlı bir yapay sinir ağı oluşturup eğitmek için gereken temel kod:

```python
import numpy as np
import newton_vision as nv

# 1. Modeli Tanımla
model = nv.Sequential(
    nv.Linear(in_features=10, out_features=32),
    nv.ReLU(),
    nv.Linear(in_features=32, out_features=2)
)

# 2. Kayıp Fonksiyonu ve Optimizatör
criterion = nv.gravity.MSELoss()
optimizer = nv.gravity.Adam(model.parameters(), lr=0.01)

# 3. Veri
x = nv.Tensor(np.random.randn(8, 10), requires_grad=True)
y = nv.Tensor(np.random.randn(8, 2))

# 4. Eğitim Adımı
optimizer.zero_grad()          # 1. Önceki gradyanları sıfırla
pred = model(x)                # 2. İleri besleme (Forward Pass)
loss = criterion(pred, y)      # 3. Hatayı hesapla
loss.backward()                # 4. Geriye yayılım (Backward Pass)
optimizer.step()               # 5. Ağırlıkları güncelle

print(f"Adım 1 Kayıp: {float(loss.data):.6f}")
```

---

## 3. Tensör & Otomatik Türev (Autograd Engine)

### `nv.Tensor`
Framework'ün temel veri yapısıdır. NumPy dizilerini sarmalar ve türev grafını tutar.

```python
x = nv.Tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 5.0
loss = y.sum()
loss.backward()

print("x gradyanı:", x.grad)  # [4.0, 6.0] (y' = 2x)
```

### `nv.no_grad()` Context Manager
Çıkarım (inference) ve test sırasında türev grafının kurulmasını engelleyerek bellek tasarrufu ve maksimum hız sağlar:

```python
with nv.no_grad():
    predictions = model(x)  # requires_grad=False, graph oluşturulmaz
```

---

## 4. Katmanlar (Layers API)

Tüm katmanlar `nv.Module` sınıfından türetilmiştir ve `parameters()`, `to()`, `train()`, `eval()` yaşam döngüsünü destekler.

| Katman | Açıklama | Kullanım Örneği |
|---|---|---|
| `nv.Linear` | Tam bağlantılı (Dense) katman | `nv.Linear(in_features=128, out_features=10)` |
| `nv.Conv2d` | 2 Boyutlu Evrişim Katmanı | `nv.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)` |
| `nv.BatchNorm2d` | 2D Yığın Normalizasyonu | `nv.BatchNorm2d(num_features=16)` |
| `nv.LayerNorm` | Katman Normalizasyonu (Transformer) | `nv.LayerNorm(normalized_shape=768)` |
| `nv.ReLU` | Düzeltilmiş Doğrusal Birim ($max(0, x)$) | `nv.ReLU()` |
| `nv.LeakyReLU` | Sızıntılı ReLU aktivasyonu | `nv.LeakyReLU(negative_slope=0.01)` |
| `nv.Sigmoid` | Lojistik aktivasyon ($1 / (1 + e^{-x})$) | `nv.Sigmoid()` |
| `nv.Tanh` | Hiperbolik tanjant aktivasyonu | `nv.Tanh()` |
| `nv.GELU` | Gaussian Error Linear Unit (ViT/BERT) | `nv.GELU()` |
| `nv.MaxPool2d` | 2D Maksimum Havuzlama | `nv.MaxPool2d(kernel_size=2, stride=2)` |
| `nv.AvgPool2d` | 2D Ortalama Havuzlama | `nv.AvgPool2d(kernel_size=7, stride=1)` |
| `nv.MultiHeadAttention` | Çok Başlı Öz-Dikkat (MHSA) | `nv.MultiHeadAttention(embed_dim=256, num_heads=8)` |
| `nv.Flatten` | Eksenleri düzleştirir; autograd grafını korur | `nv.Flatten()` |
| `nv.Conv1d` | 1 Boyutlu Evrişim `(N, C, L)` | `nv.Conv1d(16, 32, kernel_size=3, padding=1)` |
| `nv.BatchNorm1d` | `(N, C)` / `(N, C, L)` yığın normalizasyonu | `nv.BatchNorm1d(num_features=32)` |
| `nv.Embedding` | Tam sayı indeks → öğrenilebilir vektör | `nv.Embedding(1000, 64, padding_idx=0)` |
| `nv.Upsample` | Büyütme: nearest / linear / bilinear | `nv.Upsample(scale_factor=2, mode="bilinear")` |

> ⚠️ `nv.Tensor(x.data.reshape(...))` ile elle düzleştirmek autograd grafını koparır ve önceki katmanlar eğitilmez. `nv.Flatten()` veya `x.reshape(...)` kullanın.

---

## 5. Model Konteynırları (Sequential & ModuleList)

### `nv.Sequential`
Katmanları sıralı zincirleme çalıştırmak için kullanılır:
```python
model = nv.Sequential(
    nv.Conv2d(1, 16, 3, padding=1),
    nv.BatchNorm2d(16),
    nv.ReLU(),
    nv.MaxPool2d(2)
)
out = model(x)
```

### `nv.ModuleList`
Alt modülleri bir Python listesi gibi saklayan ve model parametrelerine otomatik kaydeden konteynır:
```python
class MyBlock(nv.Module):
    def __init__(self):
        super().__init__()
        self.layers = nv.ModuleList([nv.Linear(64, 64) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

---

## 6. Veri Kümeleri ve Yükleyici (Dataset & DataLoader)

Framework 3 temel veri kümesi formatını yerel (native) olarak destekler:

### A. Bellekteki Diziler (`nv.TensorDataset`)
NumPy dizilerini veya Tensörleri doğrudan mini-batch'lere böler:
```python
dataset = nv.TensorDataset(X_data, y_labels)
loader = nv.DataLoader(dataset, batch_size=32, shuffle=True)
```

### B. Resim Klasörleri (`nv.ImageFolder`) - Torchvision Standardı
Görüntü dosyalarını (`.jpg`, `.png`, `.bmp`) alt klasör isimlerine göre otomatik olarak etiketler:
```
dataset_dir/
    kedi/  (kedi1.jpg, kedi2.png) -> 0. sınıf
    kopek/ (kopek1.jpg, kopek2.png) -> 1. sınıf
```
```python
dataset = nv.ImageFolder("dataset_dir/", target_size=(28, 28), grayscale=True)
print("Sınıflar:", dataset.classes)  # ['kedi', 'kopek']
loader = nv.DataLoader(dataset, batch_size=32, shuffle=True)
```

### C. Tabüler / CSV Verileri (`nv.CSVDataset`)
Excel, Pandas veya CSV tablolarındaki sayısal öznitelikleri doğrudan tensörlere dönüştürür:
```python
dataset = nv.CSVDataset("veriler.csv", target_column="hedef_etiket")
loader = nv.DataLoader(dataset, batch_size=64, shuffle=True)
```

---

## 6.1 Hata Yönetimi ve Güvenlik Kalkanları (Error Handling & Guards)

Sistem, çalışma anında çökmeleri (segmentation fault) ve sessiz hataları önleyen sıkı korumalara sahiptir:

1. **Boyut ve Şekil Denetimi (Shape Guards):**
   * Bir `nv.Linear(10, 2)` katmanına 10 yerine 12 boyutunda bir tensör verilirse, C++ çekirdeği çökmez; açıklayıcı bir `ValueError` fırlatılır:
     `[Linear Boyut Hatası] Giriş tensörünün son boyutu (12) in_features (10) ile eşleşmiyor!`
   * `nv.Conv2d` katmanına 4 boyutlu olmayan veya kanal sayısı uyuşmayan bir tensör verildiğinde anında açık hata mesajı döner.
2. **Cihaz Uyuşmazlığı Kalkanı (Device Guard):**
   * CPU tensörü ile CUDA tensörü toplanmaya veya çarpılmaya çalışıldığında `RuntimeError` ile işlem durdurulur, bellek bozulması önlenir.
3. **Gradyan Güvenliği:**
   * Skaler olmayan bir çıktı üzerinden argümansız `backward()` çağrıldığında sistem PyTorch gibi uyarır.
   * `no_grad()` açıkken gradyan hesaplamaya çalışılırsa gereksiz bellek tahsisatı yapılmaz.

---

## 7. Kayıp Fonksiyonları & Optimizatörler (Gravity Engine)

### Kayıp Fonksiyonları (`nv.gravity`)
* **`nv.gravity.CrossEntropyLoss(reduction='mean')`**: Sınıflandırma görevleri için numerik kararlı Softmax + NLLLoss. Hem 1D sınıf indekslerini (`[0, 2, 1]`) hem 2D One-Hot etiketleri destekler.
* **`nv.gravity.MSELoss(reduction='mean')`**: Regresyon görevleri için ortalama karesel hata. `reduction='mean'`, `'sum'` veya `'none'` parametrelerini alır.
* **`nv.L1Loss`**, **`nv.BCELoss`** (girdi olasılık), **`nv.BCEWithLogitsLoss`** (girdi logit, sayısal kararlı): PyTorch ile aynı değer ve gradyanlar.

### Optimizatörler (`nv.gravity`)
* **`nv.gravity.Adam(model.parameters(), lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)`**: Adaptif moment kestirimi optimizatörü.
* **`nv.gravity.SGD(model.parameters(), lr=0.01, momentum=0.9)`**: Momentum destekli stokastik gradyan inişi.
* **`nv.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)`**: Ayrık ağırlık azaltmalı Adam (PyTorch AdamW ile aynı).
* Tüm optimizatörlerde **`state_dict()` / `load_state_dict()`** var; kaldığınız yerden aynı sonuçla devam edebilirsiniz.

### Eğitim Yardımcıları
* **`nv.StepLR(optimizer, step_size, gamma)`**, **`nv.CosineAnnealingLR(optimizer, T_max, eta_min)`**: Her epoch sonunda `scheduler.step()` çağrılır.
* **`nv.clip_grad_norm_(model.parameters(), max_norm)`**: `backward()` ile `step()` arasında çağrılır; kırpmadan önceki normu döndürür.
* **`nv.transforms`**: `Compose`, `RandomCrop(size, padding)`, `RandomHorizontalFlip(p)`, `Normalize(mean, std)`. Kullanım: `nv.TensorDataset(x, y, transform=...)`.
* **`nv.text.Tokenizer(level="word" | "char")`**: `fit(metinler)`, `encode_batch(metinler, max_len)` → `nv.Embedding`'e verilecek `(N, L)` int64 matris.

---

## 8. Hazır Modeller (Model Zoo)

`newton_vision.models` modülü üzerinden önceden tanımlanmış standart mimarileri doğrudan kullanabilirsiniz:

```python
# 1. ResNet-18 (Residual Networks)
resnet = nv.models.resnet18(num_classes=1000)

# 2. MobileNetV2 (Inverted Residuals & Depthwise Separable Convs)
mobilenet = nv.models.mobilenet_v2(num_classes=1000)

# 3. Vision Transformer (ViT-Tiny & ViT-Base)
vit = nv.models.vit_tiny(img_size=224, patch_size=16, num_classes=1000)
```

---

## 9. PyTorch Ağırlık Aktarımı (State Dict)

PyTorch `state_dict`'ini NumPy dizilerine çevirip **aynı mimarideki ve aynı parametre isimlerini kullanan** bir newton-vision modeline yükleyebilirsiniz. Doğrusal katman ağırlıkları (`out_features, in_features` -> `in_features, out_features`) otomatik olarak transpoze edilir.

> **Davranış:**
> - Şekli uyuşmayan anahtar `ValueError` verir; bu durumda model değişmeden kalır.
> - Eşleşmeyen veya eksik anahtar varsa `UserWarning` verilir; `strict=True` ile `RuntimeError`.
> - Fonksiyon `LoadResult(missing_keys, unexpected_keys)` döndürür.
> - torchvision ResNet-18 aktarımı doğrulanmıştır: max fark 6e-6, kosinüs 1.0.

```python
import torch
import torchvision.models as tv_models
import newton_vision as nv

# 1. PyTorch modelini ve state_dict'ini al
pt_resnet = tv_models.resnet18(weights=None)
pt_weights = {k: v.numpy() for k, v in pt_resnet.state_dict().items()}

# 2. newton-vision modeline aktar
nv_resnet = nv.models.resnet18(num_classes=1000)
result = nv.load_state_dict(nv_resnet, pt_weights)
print(result.missing_keys, result.unexpected_keys)  # [] [] beklenir
```

---

## 10. INT8 Kuantizasyon

Modelin FP32 ağırlıklarını **INT8**'e dönüştürerek ağırlık belleğini **~%75 düşürür** (≈4 kat küçülme). Bu bir bellek optimizasyonudur; çıkarım hızı artmaz, ölçümlerde biraz düşer.

```python
model = nv.models.resnet18(num_classes=1000)

# Modelin Linear/Conv2d katmanlarını yerinde (in-place) INT8 karşılıklarıyla değiştirir,
# istatistik sözlüğü döndürür
stats = nv.quantize_model(model)

print(f"FP32 Bellek: {stats['fp32_memory_kb']:.2f} KB")
print(f"INT8 Bellek: {stats['int8_memory_kb']:.2f} KB")
print(f"Tasarruf  : %{stats['memory_reduction_percent']:.2f}")
```

---

## 11. Model Kaydetme & Yükleme (JIT / Serialization)

Eğitilmiş bir modelin durumunu (parametreler, bias'lar, running mean/var) diskte sıkıştırılmış ikili formatta saklayabilirsiniz:

```python
# Diske kaydet (.nv dosyası)
nv.save_model(model, "models/my_trained_model.nv")

# Diskten geri yükle
new_model = MyModel()
nv.load_model(new_model, "models/my_trained_model.nv")
new_model.eval()
```

---

## 12. Somut Uygulamalar ve CLI Araçları

Framework ile birlikte gelen 3 hazır somut araç:

### 1. Siber Güvenlik Zararlı Yazılım Tespiti (`apps/malware_classifier.py`)
Bir dosyanın ilk 1024 byte'ını 32x32 gri tonlu görüntüye çevirip CNN ile [TEMIZ] / [ZARARLI] sınıflandırması yapan bir **demo**. Model **sentetik** veriyle eğitilir (zararlı = NOP sled + rastgele byte); gerçek zararlı yazılımlar üzerindeki başarısı ölçülmemiştir, güvenlik aracı olarak kullanılmamalıdır:
```powershell
python apps/malware_classifier.py --scan C:\Windows\explorer.exe
```

### 2. Canlı Görsel Sınıflandırma CLI (`apps/image_predictor.py`)
Masaüstünüzdeki herhangi bir `.png` / `.jpg` görselini açar, terminale ASCII önizlemesini basar ve milisaniyeler içinde sınıflandırır:
```powershell
python apps/image_predictor.py --image assets/real_mnist/number.jpg
```

### 3. PyTorch vs newton-vision Benchmark (`apps/pytorch_int8_benchmark.py`)
Sisteminizdeki PyTorch kütüphanesi ile `newton-vision`'ın matematiksel çıktısını ve INT8 hızını kıyaslar:
```powershell
python apps/pytorch_int8_benchmark.py
```
