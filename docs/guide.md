# Kullanım Rehberi

newton-vision'ın tüm API'si: tensörler ve autograd, katmanlar, modeller, eğitim, veri, kuantizasyon, serileştirme/ONNX, PyTorch ağırlık aktarımı, cihazlar ve C++ API'si. Kısa giriş için [README](../README.md).

## 🗺️ Modül Haritası & API Referansı

`newton-vision` Newton fiziği temalı bir isimlendirme sistemine sahiptir:

```
newton_vision
├── optics         → Görsel Katmanlar (Conv2d, Linear, BatchNorm2d, LayerNorm, GELU, MultiHeadAttention...)
├── fluxion        → Autograd Türev Motoru (Tensor, backward, .cuda(), .cpu(), .to(device))
├── gravity        → Kayıp & Optimizatörler (MSELoss, CrossEntropyLoss, SGD, Adam)
├── principia      → Cihaz Yönetimi (is_cuda_available, get_device, set_device)
├── models         → Model Zoo (ResNet18, MobileNetV2, VisionTransformer, vit_tiny, vit_base)
├── quantization   → INT8 Kuantizasyon (quantize_model, quantize_tensor, QuantizedLinear)
├── weights        → PyTorch Ağırlık Aktarıcı (load_state_dict)
└── jit            → Serileştirme & Çıkarım (InferenceEngine, save/load_model, export_onnx)
```

Her modüle `import newton_vision as nv` ile erişilir:

```python
import newton_vision as nv

# Katmanlar (optics aracılığıyla veya doğrudan):
nv.Conv2d, nv.Linear, nv.ReLU, nv.GELU, nv.BatchNorm2d, nv.LayerNorm, nv.MultiHeadAttention

# Modeller:
nv.models.resnet18(), nv.models.mobilenet_v2(), nv.models.vit_tiny(), nv.models.vit_base()

# Eğitim:
nv.gravity.CrossEntropyLoss(), nv.gravity.Adam(), nv.gravity.SGD()

# Kuantizasyon:
nv.quantize_model(), nv.quantize_tensor(), nv.QuantizedLinear()

# JIT / Serileştirme:
nv.save_model(), nv.load_model(), nv.export_onnx(), nv.InferenceEngine()

# Cihaz:
nv.principia.is_cuda_available(), nv.principia.set_device("cuda")
```

---

## 🔬 Tensor & Autograd Motoru (Fluxion)

### Tensor Oluşturma

```python
import newton_vision as nv
import numpy as np

# NumPy dizisinden
x = nv.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)

# Çok boyutlu
x = nv.Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32), requires_grad=True)
```

> **Cihaz notu:** Tensör verisi her zaman host belleğinde (NumPy) tutulur; `device` alanı yalnızca hangi C++ çekirdeğinin çağrılacağını belirler. CUDA ile derlenmiş bir build'de `device="cuda"` olan tensörler için `conv2d`, `relu` ve `maxpool2d` forward işlemleri GPU kernel'larında çalışır (her çağrıda host ↔ device kopyası yapılır); diğer tüm işlemler ve tüm backward hesapları CPU'dadır.

### Temel Özellikler

```python
x = nv.Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)

print(x.shape)     # (2, 3, 4)
print(x.ndim)      # 3
print(x.strides)   # (48, 16, 4) — byte cinsinden
print(x.device)    # "cpu"
print(x.data)      # NumPy ndarray'e doğrudan erişim
print(x.grad)      # Gradyan (backward sonrası dolar)
```

### Desteklenen Operasyonlar (Autograd Destekli)

```python
a = nv.Tensor(np.array([2.0, 3.0]), requires_grad=True)
b = nv.Tensor(np.array([4.0, 5.0]), requires_grad=True)

c = a + b        # Toplama (backward destekli)
d = a * b        # Çarpma (backward destekli)
e = a - b        # Çıkarma (backward destekli)
f = a.sum()      # Toplam skaler (backward destekli)
g = a.mean()     # Ortalama skaler (backward destekli)
h = a[0]         # İndeksleme / dilimleme (backward destekli)
```

### Backward — Otomatik Türev Hesaplama

```python
x = nv.Tensor(np.array([2.0, 3.0]), requires_grad=True)
w = nv.Tensor(np.array([0.5, -1.0]), requires_grad=True)

# Forward
y = (x * w).sum()

# Backward — tüm requires_grad=True tensörlerin gradyanlarını hesaplar
y.backward()

print(x.grad)  # [0.5, -1.0] — ∂y/∂x = w
print(w.grad)  # [2.0,  3.0] — ∂y/∂w = x
```

**Autograd Mekanizması:**
1. Her operasyon bir çocuk düğümü (`_children`) ve bir `_backward` fonksiyonu kaydeder
2. `backward()` çağrıldığında topolojik sıralama (topological sort) ile tüm graf tersine gezilir
3. Zincir kuralı (chain rule) ile her düğümün gradyanı hesaplanır

### Cihaz Transferi

```python
x = nv.Tensor(np.zeros((2, 3)), device="cpu")

# CPU → CUDA (CUDA build'i yoksa RuntimeError fırlatır)
x_gpu = x.cuda()       # veya x.to("cuda")

# CUDA → CPU
x_cpu = x_gpu.cpu()     # veya x_gpu.to("cpu")
```

> `.cuda()` veriyi GPU belleğine taşımaz; yalnızca tensörün cihaz etiketini değiştirir (bkz. yukarıdaki cihaz notu). GitHub Actions ve varsayılan pip/MinGW build'i CUDA'sızdır. CUDA build'i için bkz. **Cihaz Yönetimi → CUDA build'i**.

---

## 🧱 Katman Referansı — Tüm Layer API

### Conv2d — 2D Evrişim Katmanı

```python
conv = nv.Conv2d(
    in_channels=3,       # Giriş kanal sayısı
    out_channels=64,     # Çıkış kanal sayısı (filtre sayısı)
    kernel_size=3,       # Filtre boyutu (3×3)
    stride=1,            # Kayma adımı
    padding=1,           # Kenar dolgusu
    dilation=1,          # Dilated convolution genişlemesi
    groups=1,            # Grouped convolution (depthwise için groups=in_channels)
    bias=True,           # Bias vektörü kullanılsın mı
    device="cpu"         # "cpu" veya "cuda"
)

# İleri geçiş
x = nv.Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32), requires_grad=True)
y = conv(x)  # (1, 64, 32, 32) — padding=1 ile boyut korunur

# Autograd destekli — backward() otomatik çalışır
y.sum().backward()
print(x.grad.shape)  # (1, 3, 32, 32)
```

**C++ Backend'de `im2col + GEMM` ile çalışır.** 2D görüntüyü sütun matrisine dönüştürerek BLAS-tarzı matris çarpımı yapar. Bu yöntem iç içe döngülü naif evrişime göre **~10x hızlanma** sağlar.

### Linear — Tam Bağlantılı Katman

```python
fc = nv.Linear(in_features=512, out_features=10, bias=True, device="cpu")
x = nv.Tensor(np.random.randn(4, 512).astype(np.float32), requires_grad=True)
y = fc(x)  # (4, 10)
```

Matematik: `y = x @ W + b` (Kaiming He başlatma ile)

### Aktivasyon Fonksiyonları

```python
relu = nv.ReLU()          # max(0, x)
leaky = nv.LeakyReLU(0.01)  # x if x > 0 else 0.01*x
sigmoid = nv.Sigmoid()     # 1 / (1 + exp(-x))
tanh = nv.Tanh()           # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
gelu = nv.GELU()           # x * Φ(x) — Transformer'ların standart aktivasyonu
```

Tüm aktivasyonlar C++ backend ile hızlandırılmıştır. GELU ve ReLU autograd backward desteğine sahiptir.

### BatchNorm2d — Batch Normalizasyonu

```python
bn = nv.BatchNorm2d(num_features=64, eps=1e-5, device="cpu")
x = nv.Tensor(np.random.randn(8, 64, 16, 16).astype(np.float32))
y = bn(x)  # (8, 64, 16, 16) — normalize + scale + shift
```

Running mean ve running variance takibi ile eğitim sırasında istatistik toplama.

### LayerNorm — Katman Normalizasyonu

```python
ln = nv.LayerNorm(normalized_shape=192, eps=1e-5, device="cpu")
x = nv.Tensor(np.random.randn(4, 197, 192).astype(np.float32))
y = ln(x)  # (4, 197, 192) — son boyut üzerinden normalize
```

Transformer mimarilerinde her bloktan önce kullanılır.

### MultiHeadAttention — Çok Başlı Öz-Dikkat

```python
mha = nv.MultiHeadAttention(embed_dim=192, num_heads=3, bias=True, device="cpu")
x = nv.Tensor(np.random.randn(2, 197, 192).astype(np.float32))
y = mha(x)  # (2, 197, 192)
```

**İç Çalışma Mekanizması:**
1. Girişi Q, K, V projeksiyonlarına ayırır (3 ayrı Linear katman)
2. Her başlık (head) için `head_dim = embed_dim / num_heads` boyutunda bağımsız attention hesaplar
3. `Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V`
4. Tüm başlıkları birleştirip çıkış projeksiyonu uygular

### MaxPool2d & AvgPool2d — Havuzlama Katmanları

```python
maxpool = nv.MaxPool2d(kernel_size=2, stride=2)            # padding=0 (varsayılan)
maxpool_pad = nv.MaxPool2d(kernel_size=3, stride=2, padding=1)  # PyTorch ile aynı -inf dolgu; padding <= kernel_size // 2
avgpool = nv.AvgPool2d(kernel_size=7, stride=1)

x = nv.Tensor(np.random.randn(1, 64, 32, 32).astype(np.float32), requires_grad=True)
y_max = maxpool(x)  # (1, 64, 16, 16) — boyutu yarıya indirir
y_avg = avgpool(x)  # (1, 64, 26, 26)
```

MaxPool2d autograd backward desteğine sahiptir.

### Flatten, Conv1d, BatchNorm1d, Embedding, Upsample

Hepsi PyTorch ile aynı adlara, ağırlık düzenine ve semantiğe sahiptir. Forward ve backward, `tests/test_training_components.py` içinde PyTorch float64 referansıyla doğrulanır.

```python
flat = nv.Flatten()                       # (N, C, H, W) -> (N, C*H*W); autograd grafını korur
conv1d = nv.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)  # (N, C, L)
bn1d = nv.BatchNorm1d(32)                 # (N, C) veya (N, C, L)
emb = nv.Embedding(1000, 64, padding_idx=0)   # tam sayı indeks -> vektör (ağırlık: emb.weight)
up = nv.Upsample(scale_factor=2)          # mode: "nearest" | "linear" (N,C,L) | "bilinear" (N,C,H,W); tam sayı ölçek
up_bl = nv.Upsample(2, mode="bilinear", align_corners=False)
```

> **Düzleştirme uyarısı:** `nv.Tensor(x.data.reshape(N, -1))` yeni bir yaprak tensör oluşturur ve autograd grafını **koparır**; önceki katmanlar eğitilmez. Her zaman `nv.Flatten()` veya `x.reshape(N, -1)` kullanın.

---

## 🧩 Model Konteynırları (Sequential & ModuleList)

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

## 🏗️ Model Zoo — Hazır Mimariler

### ResNet-18

```python
model = nv.models.resnet18(num_classes=10)
x = nv.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
logits = model(x)  # (1, 10)
```

**Mimari:** `Conv2d(7×7) → BN → ReLU → MaxPool → 4×[ResidualBlock(3×3)] → AvgPool → Linear`

ResidualBlock skip-connection ile gradient vanishing sorununu çözer:
```
input ──→ Conv → BN → ReLU → Conv → BN ──→ (+) → ReLU → output
  │                                          ↑
  └──────────── identity shortcut ───────────┘
```

### MobileNetV2

```python
model = nv.models.mobilenet_v2(num_classes=10)
x = nv.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
logits = model(x)  # (1, 10)
```

**Mimari:** Inverted Residual Block yapısı — depthwise separable convolution ile parametre verimliliği:
```
input → 1×1 Conv(expand) → BN → ReLU → 3×3 DWConv → BN → ReLU → 1×1 Conv(project) → BN → (+) → output
```

### Vision Transformer (ViT)

```python
# ViT-Tiny (5.7M parametre)
model_tiny = nv.models.vit_tiny(num_classes=1000)

# ViT-Base (86M parametre)
model_base = nv.models.vit_base(num_classes=1000)

# Özel yapılandırma
from newton_vision.models.vit import VisionTransformer
custom_vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=100,
    embed_dim=384,
    depth=6,
    num_heads=6,
    mlp_ratio=4.0
)
```

**ViT Çalışma Prensibi (adım adım):**

```
1. PatchEmbedding: 224×224 görüntüyü 16×16 patch'lere böl → 196 patch
2. Her patch'i Conv2d(16×16, stride=16) ile embed_dim boyutuna projekte et
3. [CLS] token'ı dizinin başına ekle → 197 token
4. Positional Embedding ekle (öğrenilebilir)
5. 12× TransformerBlock:
   ├── LayerNorm → MultiHeadAttention → Residual Connection
   └── LayerNorm → MLP(Linear→GELU→Linear) → Residual Connection
6. Son LayerNorm
7. [CLS] token çıktısını al → Linear Head → sınıf logits
```

**ViT Yapılandırma Karşılaştırması:**

| Yapılandırma | Embed Dim | Depth | Heads | MLP Ratio | Parametre |
|---|---|---|---|---|---|
| ViT-Tiny | 192 | 12 | 3 | 4.0 | ~5.7M |
| ViT-Base | 768 | 12 | 12 | 4.0 | ~86M |

---

## 📉 Kayıp Fonksiyonları & Optimizatörler (Gravity)

### CrossEntropyLoss — Çapraz Entropi Kaybı

```python
loss_fn = nv.gravity.CrossEntropyLoss()

logits = nv.Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
labels = np.array([3, 7, 1, 0])  # Hedef sınıf indeksleri

loss = loss_fn(logits, labels)
loss.backward()  # Gradyanlar otomatik hesaplanır
```

Dahili olarak numerik kararlı softmax uygular: `exp(x - max(x))`

### MSELoss — Ortalama Kare Hata

```python
loss_fn = nv.gravity.MSELoss()

pred = nv.Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
target = nv.Tensor(np.array([1.5, 2.5, 3.5]))

loss = loss_fn(pred, target)
loss.backward()
```

### Adam Optimizatörü

```python
optimizer = nv.gravity.Adam(
    params=model.parameters(),  # Eğitilebilir parametre listesi
    lr=0.001,                   # Öğrenme oranı
    beta1=0.9,                  # 1. moment üstel azalma faktörü
    beta2=0.999,                # 2. moment üstel azalma faktörü
    eps=1e-8                    # Sayısal kararlılık epsilon değeri
)

# Eğitim adımı
optimizer.zero_grad()   # Gradyanları sıfırla
loss.backward()         # Gradyanları hesapla
optimizer.step()        # Parametreleri güncelle
```

Adam, bias-corrected first & second moment estimation uygular:
```
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)
θ = θ - lr · m̂ / (√v̂ + ε)
```

### SGD Optimizatörü (Momentum Destekli)

```python
optimizer = nv.gravity.SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9   # 0 = klasik SGD, > 0 = momentum destekli
)
```

### Ek Kayıplar: L1Loss, BCELoss, BCEWithLogitsLoss

```python
nv.L1Loss()             # |pred - target|
nv.BCELoss()            # girdi olasılık [0, 1]; log değerleri -100'de kırpılır (PyTorch gibi)
nv.BCEWithLogitsLoss()  # girdi logit; sayısal kararlı (büyük logit'lerde inf/nan üretmez)
```

### LR Scheduler, Gradyan Kırpma, Eğitime Devam

```python
optimizer = nv.Adam(model.parameters(), lr=1e-3)   # veya nv.AdamW(..., weight_decay=0.01): ayrık ağırlık azaltma
scheduler = nv.StepLR(optimizer, step_size=10, gamma=0.5)   # veya nv.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(epochs):
    for bx, by in loader:
        loss = loss_fn(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        nv.clip_grad_norm_(model.parameters(), max_norm=1.0)  # kırpmadan önceki normu döndürür
        optimizer.step()
    scheduler.step()

# Kontrol noktası: model + optimizer (+ scheduler) durumu
ckpt = {"model": model.state_dict(), "opt": optimizer.state_dict(), "sched": scheduler.state_dict()}
# ... devam ederken:
model.load_state_dict(ckpt["model"], strict=True)
optimizer.load_state_dict(ckpt["opt"])      # parametre sayısı/şekli uymazsa ValueError
scheduler.load_state_dict(ckpt["sched"])
```

`AdamW`, `StepLR`, `CosineAnnealingLR` ve `clip_grad_norm_` PyTorch ile aynı değerleri üretir. Kesilip `state_dict`'ten devam ettirilen eğitim, kesintisiz eğitimle aynı ağırlıklara ulaşır (testlerle doğrulandı).

### Veri Artırma (`nv.transforms`)

```python
train_tf = nv.transforms.Compose([
    nv.transforms.RandomCrop(28, padding=2),
    nv.transforms.RandomHorizontalFlip(p=0.5),
    nv.transforms.Normalize(mean=[0.1307], std=[0.3081]),
])
dataset = nv.TensorDataset(x_train, y_train, transform=train_tf)  # her örneğe ayrı uygulanır
```

Dönüşümler tek bir `(C, H, W)` NumPy örneği üzerinde çalışır. `TensorDataset(..., transform=...)` newton-vision'a özgü bir kolaylıktır; PyTorch'un TensorDataset'inde bu parametre yoktur.

### Metin: Tokenizer (`nv.text`)

```python
tok = nv.text.Tokenizer(level="word", min_freq=1).fit(train_texts)   # veya level="char"
ids = tok.encode_batch(batch_texts, max_len=64)    # (N, 64) int64, <pad>=0 ile dolgulu, <unk>=1
emb = nv.Embedding(tok.vocab_size, 128, padding_idx=tok.pad_id)
x = emb(ids)                                        # (N, 64, 128)
tok.decode(ids[0])                                  # özel tokenlar atlanır
```

Basit kelime/karakter tokenizer'ıdır. Kelime dağarcığı deterministiktir (sıklık, sonra alfabetik) ve `state_dict()` ile kaydedilir. Alt-kelime (BPE/WordPiece) yöntemleri yoktur.

---

## 📦 Veri Kümeleri ve Yükleyici (Dataset & DataLoader)

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

## 🛡️ Hata Yönetimi ve Güvenlik Kalkanları (Error Handling & Guards)

Sistem, çalışma anında çökmeleri (segmentation fault) ve sessiz hataları önleyen sıkı korumalara sahiptir:

1. **Boyut ve Şekil Denetimi (Shape Guards):**
   * Bir `nv.Linear(10, 2)` katmanına son boyutu 12 olan bir tensör verilirse, beklenen ve gelen boyutu belirten bir `ValueError` (`[Linear Boyut Hatası] ...`) fırlatılır.
   * `nv.Conv2d` katmanına 4 boyutlu olmayan veya kanal sayısı uyuşmayan bir tensör verildiğinde anında açık hata mesajı döner.
2. **Cihaz Uyuşmazlığı Kalkanı (Device Guard):**
   * CPU tensörü ile CUDA tensörü toplanmaya veya çarpılmaya çalışıldığında `RuntimeError` ile işlem durdurulur, bellek bozulması önlenir.
3. **Gradyan Güvenliği:**
   * Skaler olmayan bir çıktı üzerinden argümansız `backward()` çağrıldığında, PyTorch gibi `RuntimeError` fırlatılır.
   * `no_grad()` açıkken yeni tensörler grafa bağlanmaz ve gradyan tamponu ayrılmaz.
4. **Ağırlık Yükleme:** Şekli uyuşmayan anahtarlar `ValueError` verir; `strict=True` ile eksik/beklenmeyen anahtarlar `RuntimeError` verir (bkz. PyTorch Ağırlık Aktarımı).
5. **ONNX Dışa Aktarımı:** Dönüştürücüsü olmayan katmanlar `NotImplementedError`, onnxruntime doğrulamasındaki uyuşmazlıklar `RuntimeError` verir.

---

## 🧊 INT8 Kuantizasyon Motoru

### Neden Kuantizasyon?

FP32 ağırlıkları INT8'e dönüştürerek:
- **~4x ağırlık bellek tasarrufu** (4 byte → 1 byte per parametre, artı ölçek değerleri)
- **Düşük doğruluk kaybı** (Per-Channel kuantizasyon ile)

**Hız hakkında dürüst not:** Mevcut sürümde INT8 kuantizasyonu bir **bellek** optimizasyonudur, hız optimizasyonu değildir. `QuantizedConv2d` her forward'da ağırlıkları FP32'ye geri çevirip FP32 evrişim çalıştırır; `QuantizedLinear` ise INT32 birikimli matris çarpımını NumPy ile yapar. C++ tarafındaki `gemm_int8_cpu` henüz Python'a bağlı değildir. Ölçümlerde INT8 model FP32'den biraz daha yavaştır (bkz. [docs/performance.md](performance.md)).

### Tensör Düzeyinde Kuantizasyon

```python
import newton_vision as nv
import numpy as np

data = np.random.randn(4, 4).astype(np.float32)

# ── Global Kuantizasyon ───────────────────────────────────────────
q_int8, scale = nv.quantize_tensor(data)
print(f"INT8 veri tipi: {q_int8.dtype}")   # int8
print(f"Scale faktörü: {scale:.6f}")

# ── Geri Dönüştürme (Dequantize) ─────────────────────────────────
recovered = nv.dequantize_tensor(q_int8, scale)
error = np.mean(np.abs(data - recovered))
print(f"Ortalama kuantizasyon hatası: {error:.6f}")  # Tipik: < 0.02

# ── Per-Channel Kuantizasyon ──────────────────────────────────────
q_pc, scales = nv.quantize_tensor_per_channel(data, axis=0)
print(f"Kanal bazlı scale vektörü: {scales}")
```

**Per-Channel vs Global Kuantizasyon:**
- **Global:** Tüm tensör için tek bir `scale` değeri. Aşırı uç değerler (outliers) tüm kanalların hassasiyetini bozar.
- **Per-Channel:** Her filtre/kanal için bağımsız `scale_i` değeri. Outlier'lar sadece kendi kanallarını etkiler. **Her zaman Per-Channel tercih edin.**

### Model Düzeyinde Kuantizasyon

```python
model = nv.models.resnet18(num_classes=10)

# Tüm Linear ve Conv2d katmanlarını INT8'e dönüştür
stats = nv.quantize_model(model, per_channel=True)

print(f"Kuantize parametre sayısı: {stats['total_quantized_params']:,}")
print(f"FP32 bellek:  {stats['fp32_memory_kb']:.1f} KB")
print(f"INT8 bellek:  {stats['int8_memory_kb']:.1f} KB")
print(f"Tasarruf:     {stats['memory_reduction_percent']:.1f}%")  # ~75%
```

### QuantizedLinear — INT8 Lineer Katman

```python
q_linear = nv.QuantizedLinear(in_features=512, out_features=10, bias=True)

# FP32 ağırlıkları yükle (otomatik INT8'e dönüşür)
w_fp32 = np.random.randn(512, 10).astype(np.float32) * 0.01
q_linear.load_fp32_weights(w_fp32)

# INT8 çıkarım
x = np.random.randn(4, 512).astype(np.float32)
output = q_linear(x)  # INT8 matris çarpımı → FP32 çıktı
```

**Dahili INT8 matris çarpımı (NumPy, INT32 birikim):**
```
Y_fp32 = (X_int8.astype(int32) @ W_int8.astype(int32)) × (scale_x × scale_w)
```

---

## 💾 JIT & Serileştirme Motoru

### Model Kaydetme & Yükleme

```python
model = nv.models.vit_tiny(num_classes=10)

# ── Format 1: .nv (Sıkıştırılmış NumPy) ──────────────────────────
nv.save_model(model, "my_model.nv")
nv.load_model(model, "my_model.nv")

# ── Format 2: .nv.bin (Saf C++ İkili) ────────────────────────────
# C++ uygulamalarında Python bağımlılığı olmadan yüklenebilir
nv.save_model_bin(model, "my_model.nv.bin")
nv.load_model_bin(model, "my_model.nv.bin")
```

**.nv.bin Formatı:** `NVBIN100` magic header + tensör adı + 4D boyut + FP32 veri. C++ tarafında `fread()` ile doğrudan okunabilir.

### ONNX Export

```bash
pip install newton-vision[onnx]   # onnx + onnxruntime (opsiyonel bağımlılık)
```

```python
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
nv.export_onnx(model, dummy_input, "my_model.onnx")   # verify=True (varsayılan)
```

Gerçek bir ONNX (protobuf, opset 17) dosyası üretir; onnxruntime ve Netron ile açılır.
- **Nasıl çalışır:** Model örnek girdiyle bir kez izlenir. Bilinen katmanlar tek düğüm olarak yazılır: Conv2d, Conv1d, Linear, BatchNorm1d/2d, LayerNorm, MaxPool2d, AvgPool2d, aktivasyonlar, Flatten, Dropout, Upsample ve Embedding. `forward` içindeki Tensor işlemleri de ONNX'e çevrilir: `+ - * / @`, `reshape`, `transpose`, `softmax`, `mean`, `sum`, `max`, `concatenate`, `tile`, temel dilimleme. ResNet-18, MobileNetV2 ve ViT bu yolla dışa aktarılır ve testlerle doğrulanır.
- **Doğrulama:** onnxruntime kuruluysa çıktı, iki farklı girdiyle newton-vision'a karşı karşılaştırılır; uyuşmazlıkta `RuntimeError` verilir. Bu, girdiye bağlı bir değerin `Tensor(...)` ile grafa sabit olarak gömülmesini de yakalar.
- **Sınırlar:**
  - Graf, örnek girdinin şekline sabittir (batch boyutu dahil).
  - Python kontrol akışı izleme anındaki dala sabitlenir.
  - Kuantize katmanlar (`QuantizedLinear`/`QuantizedConv2d`) desteklenmez ve `NotImplementedError` verir.
  - Dışa aktarım eval modunda yapılır; modelin modu sonra geri yüklenir.

### InferenceEngine — Canlı Çıkarım Motoru

```python
model = nv.models.vit_tiny(num_classes=1000)
engine = nv.InferenceEngine(model)

# Tek kare çıkarımı
frame = np.random.randn(1, 3, 224, 224).astype(np.float32)
output, latency_ms, fps = engine.predict(frame)
print(f"Gecikme: {latency_ms:.2f} ms | FPS: {fps:.1f}")

# 100 kare akışı simülasyonu
for i in range(100):
    frame = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output, lat, fps = engine.predict(frame)

# Ortalama istatistikler
avg_lat, avg_fps = engine.get_average_stats()
print(f"Ortalama Gecikme: {avg_lat:.2f} ms | Ortalama FPS: {avg_fps:.1f}")
```

---

## 🤝 PyTorch Ağırlık Aktarımı

### Mekanizma

`newton-vision`, PyTorch'un `state_dict()` sözlük yapısını birebir destekler:

```python
import torch
import newton_vision as nv

# ── PyTorch'ta Eğit ───────────────────────────────────────────────
torch_model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
# ... eğitim kodu ...

# ── State Dict'i Çıkar ───────────────────────────────────────────
state_dict = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}

# ── newton-vision Modeline Yükle ──────────────────────────────────
nv_model = ...  # Aynı mimaride newton-vision modeli
nv.load_state_dict(nv_model, state_dict)
```

**Parametre Eşleme:**

| PyTorch Adı | newton-vision Adı |
|---|---|
| `weight` | `w` (Tensor) |
| `bias` | `bias` (Tensor) |
| `weight` (BatchNorm) | `gamma` (Tensor) |
| `bias` (BatchNorm) | `beta` (Tensor) |
| `running_mean` | `running_mean` (ndarray) |
| `running_var` | `running_var` (ndarray) |

**Dikkat:** PyTorch Linear ağırlıkları `(out_features, in_features)` şeklindedir. newton-vision `(in_features, out_features)` kullanır; `load_state_dict` 2D ağırlıklarda şekil uyuşuyorsa transpozu otomatik yapar.

**Anahtar eşleme ve hata davranışı:**
- PyTorch adları otomatik eşlenir: `weight` → `w` / `gamma`, `bias` → `bias` / `beta`, `blocks.0` → `blocks_0`, `downsample.0` → kayıt sırasına göre 0. alt modül. `num_batches_tracked` bilinçli olarak yok sayılır.
- Şekli uyuşmayan bir anahtar her zaman `ValueError` fırlatır. Değerler yazılmadan önce tüm anahtarlar doğrulanır, yani hata durumunda model değişmeden kalır.
- `strict=True`: eksik veya beklenmeyen anahtar varsa `RuntimeError`. `strict=False` (varsayılan): `UserWarning` verir ve yüklemeye devam eder.
- Dönüş değeri PyTorch'taki gibidir: `LoadResult(missing_keys, unexpected_keys)`.
- `nv.load_model` / `nv.load_model_bin` kendi checkpoint'lerini her zaman `strict=True` ile yükler.
- Kare Linear ağırlıklarında yön şekilden anlaşılamaz; anahtar adı belirler: `weight` = PyTorch `(out, in)` düzeni (transpoz edilir), `w` = newton-vision `(in, out)` düzeni.

---

## 🖥️ Cihaz Yönetimi — CPU & CUDA

```python
import newton_vision as nv

# ── CUDA Kullanılabilirliği ───────────────────────────────────────
if nv.principia.is_cuda_available():
    print("CUDA GPU mevcut!")
    nv.principia.set_device("cuda")
else:
    print("CPU modu aktif")

# ── Mevcut Cihaz ─────────────────────────────────────────────────
print(nv.principia.get_device())  # "cpu" veya "cuda"

# ── Tensör Cihaz Transferi ────────────────────────────────────────
x = nv.Tensor(np.zeros((2, 3)), device="cpu")
x_gpu = x.cuda()
x_cpu = x_gpu.cpu()
```

CUDA build'i ve kapsamı için: [cuda.md](cuda.md).

---

## ⚙️ C++ Backend Mimarisi

`newton-vision`'ın tüm ağır hesaplama işlemleri C++17 ile yazılmıştır:

### Dosya Yapısı

```
cpp/
├── layers.h           → Tüm fonksiyon imzaları (API)
├── layers.cpp         → CPU implementasyonları (im2col, GEMM, aktivasyonlar, backward)
├── simd_gemm.h        → AVX2 SIMD ile hızlandırılmış matris çarpımı
├── cuda_layers.h      → CUDA kernel imzaları
├── cuda_layers.cu     → CUDA GPU kernel implementasyonları
└── binding.cpp        → pybind11 Python bağlama katmanı
```

### C++ ile İmplemente Edilen Fonksiyonlar

| Fonksiyon | Açıklama |
|---|---|
| `im2col_cpu()` | 2D görüntüyü sütun matrisine dönüştürür (stride, dilation, padding destekli) |
| `gemm_cpu()` | Paralel matris çarpımı C = A × B + bias |
| `conv2d()` | Forward: im2col + GEMM tabanlı 2D evrişim |
| `conv2d_backward()` | Backward: ∂L/∂x, ∂L/∂w, ∂L/∂b gradyan hesaplama |
| `relu()` | Forward: max(0, x) |
| `relu_backward()` | Backward: x > 0 ? 1 : 0 |
| `leaky_relu()` | Forward: x > 0 ? x : αx |
| `sigmoid_fn()` | Forward: 1/(1+e⁻ˣ) |
| `tanh_fn()` | Forward: tanh(x) |
| `gelu_fn()` | Forward: x·Φ(x) Gaussian Error Linear Unit |
| `maxpool2d()` | Forward: bölgede maksimum değer |
| `maxpool2d_backward()` | Backward: maksimum konumuna gradyan yönlendirme |
| `avgpool2d()` | Forward: bölgede ortalama değer |
| `batchnorm2d()` | Forward: kanal bazlı normalizasyon |
| `layernorm()` | Forward: özellik boyutunda normalizasyon |
| `has_cuda()` | CUDA GPU mevcut mu? |
| `has_avx2_hardware()` | CPU AVX2 SIMD destekliyor mu? |

### im2col + GEMM Evrişim Stratejisi

Naif 6-katlı iç içe döngü yerine:

```
Giriş (N,C,H,W) → im2col → Sütun Matrisi (C×kH×kW, H_out×W_out)
                              × Filtre Matrisi (C_out, C×kH×kW)
                              = Çıkış Matrisi (C_out, H_out×W_out)
                              → reshape → (N, C_out, H_out, W_out)
```

Bu yaklaşım evrişimi tek bir büyük matris çarpımına dönüştürerek L1/L2 önbellek yerelliğini (cache locality) iyileştirir ve AVX2 GEMM çekirdeğinin kullanılmasını sağlar. Naif döngüye göre hızlanma bu repoda ayrıca ölçülmemiştir; ayrıntılar için [`research/0001-im2col-vs-direct-conv.md`](../research/0001-im2col-vs-direct-conv.md).

---

## 📚 Örnek Projeler & Öğreticiler

Aşağıdaki dosyalar proje deposunda `examples/` ve `tutorials/` dizinlerinde bulunmaktadır.

### Örnek 1: MNIST LeNet Sınıflandırma (examples/01_image_classification_lenet.py)

LeNet-5 mimarisinde MNIST el yazısı rakam sınıflandırması:

```python
import newton_vision as nv
import numpy as np

# LeNet-5 Mimarisi
conv1 = nv.Conv2d(1, 6, kernel_size=5, padding=2)
conv2 = nv.Conv2d(6, 16, kernel_size=5)
pool = nv.MaxPool2d(2, 2)
relu = nv.ReLU()
fc1 = nv.Linear(16 * 5 * 5, 120)
fc2 = nv.Linear(120, 84)
fc3 = nv.Linear(84, 10)

# Forward Pass
def forward(x):
    x = pool(relu(conv1(x)))
    x = pool(relu(conv2(x)))
    N = x.data.shape[0]
    flat = nv.Tensor(x.data.reshape(N, -1), requires_grad=True)
    return fc3(relu(fc2(relu(fc1(flat)))))
```

### Örnek 2: PyTorch ResNet-18 Ağırlık Aktarımı (examples/02_resnet18_pytorch_weights.py)

```python
import torch
import torchvision.models as tmodels
import newton_vision as nv

# PyTorch pretrained ResNet-18
torch_resnet = tmodels.resnet18(weights=tmodels.ResNet18_Weights.DEFAULT)
state_dict = {k: v.cpu().numpy() for k, v in torch_resnet.state_dict().items()}

# newton-vision ResNet-18'e aktar
nv_resnet = nv.models.resnet18(num_classes=1000)
nv.load_state_dict(nv_resnet, state_dict)
```

> **Doğrulama (2026-10-23):** Rastgele başlatılmış (BN istatistikleri rastgeleleştirilmiş) torchvision ResNet-18 ile aktarımda 122 anahtardan 102'si yüklendi, 20 `num_batches_tracked` anahtarı bilinçli olarak yok sayıldı (`missing_keys=[]`, `unexpected_keys=[]`). `(2, 3, 224, 224)` girdide çıktı farkı en fazla `6e-6`, kosinüs benzerliği `1.0`. torchvision'ın `downsample.0/1` anahtarları newton-vision'daki `downsample.conv/bn` modüllerine otomatik eşlenir.

### Örnek 3: ViT Transformer Çıkarımı (examples/03_vit_transformer_inference.py)

```python
import newton_vision as nv
import numpy as np

model = nv.models.vit_tiny(num_classes=1000)
engine = nv.InferenceEngine(model)

for i in range(10):
    frame = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output, latency, fps = engine.predict(frame)
    print(f"Frame {i+1:02d} | {latency:.1f} ms | {fps:.1f} FPS")
```

### Örnek 4: INT8 Kuantizasyon Demosu (examples/04_int8_quantization_demo.py)

```python
import newton_vision as nv
import numpy as np

model = nv.models.resnet18(num_classes=10)

# Kuantizasyon öncesi çıkarım
x = nv.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
fp32_out = model(x)

# INT8 kuantizasyon uygula
stats = nv.quantize_model(model, per_channel=True)
int8_out = model(x)

# Karşılaştırma
diff = np.mean(np.abs(fp32_out.data - int8_out.data))
print(f"FP32 vs INT8 farkı: {diff:.6f}")
print(f"Bellek tasarrufu: {stats['memory_reduction_percent']:.1f}%")
```

### Örnek 5: ONNX Export (examples/05_onnx_export_guide.py)

```python
import newton_vision as nv
import numpy as np

model = nv.models.vit_tiny(num_classes=10)
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
nv.export_onnx(model, dummy, "vit_tiny.onnx")
```

### Öğretici 1: Tekrarlanabilir MNIST Eğitimi (tutorials/01_train_mnist.py)

Gerçek MNIST verisi (`tutorials/data/*.gz`) üzerinde, Seed=42 ile mini-batch eğitim ve ardından `nv.quantize_model` ile FP32/INT8 doğruluk karşılaştırması:

```python
model = nv.Sequential(
    nv.Conv2d(1, 8, kernel_size=3, padding=1), nv.ReLU(), nv.MaxPool2d(2, 2),
    nv.Flatten(),                      # grafı korur: gradyan konvolüsyona kadar akar
    nv.Linear(8 * 14 * 14, 10),
)
optimizer = nv.Adam(model.parameters(), lr=1e-3)
loader = nv.DataLoader(nv.TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

for epoch in range(2):
    model.train()
    for bx, by in loader:
        loss = nv.CrossEntropyLoss()(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Örnek sonuç (6000 eğitim / 1000 test görseli, 2 epoch, 2026-10-23): FP32 test doğruluğu **%89.70**, INT8 **%89.80**, ağırlık belleği 61.5 KB → 15.4 KB. Script, konvolüsyon ağırlıkları değişmezse (graf kopuksa) hata verir.

**Tam MNIST ile doğrulama (aynı tarih):** LeNet, 60 bin görsel, 3 epoch, Adam (lr=1e-3) ile 10 bin test görselinde **%98.43** doğruluğa ulaştı (bu makinede epoch başına ~80 sn; tekrar ölçümde %98.29).

### Öğretici: MNIST CNN, newton-vision ve PyTorch (tutorials/02_mnist_cnn_vs_pytorch.py)

Aynı CNN, iki framework'te aynı başlangıç ağırlıkları, aynı karıştırma ve aynı veri artırmasıyla eğitilir.
- **Kullanılan bileşenler:** BatchNorm1d/2d, AdamW, kosinüs LR, `clip_grad_norm_`.
- **Değerlendirme:** Ayrılmış doğrulama seti; test seti sadece en sonda bir kez ölçülür. Ardından INT8 ve ONNX doğruluğu da ölçülür.

```bash
python tutorials/02_mnist_cnn_vs_pytorch.py                       # 10 epoch, tam veri (~16 dk)
python tutorials/02_mnist_cnn_vs_pytorch.py --epochs 1 --n-train 8000 --no-torch   # hızlı deneme
```

Sonuçlar (test %99.42, PyTorch %99.47) ve epoch tablosu: [performance.md](performance.md). Çıktılar `tutorials/output/` klasörüne yazılır.

### Öğretici 2: 3 Aşamalı Canlı Demo (tutorials/live_3stage_demo.py)

3 aşamalı karşılaştırma: Saf Eğitim → PyTorch Aktarımı → INT8 Kuantizasyon.

```python
import newton_vision as nv
import numpy as np

print("=" * 60)
print("AŞAMA 1: Saf newton-vision Eğitimi")
print("=" * 60)

fc1 = nv.Linear(16, 32)
fc2 = nv.Linear(32, 3)
relu = nv.ReLU()
optimizer = nv.gravity.Adam(fc1.parameters() + fc2.parameters(), lr=0.01)
loss_fn = nv.gravity.CrossEntropyLoss()

X = np.random.randn(50, 16).astype(np.float32)
y = np.random.randint(0, 3, 50)

for ep in range(20):
    logits = fc2(relu(fc1(nv.Tensor(X, requires_grad=True))))
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Son Loss: {float(loss.data):.4f}")

print("\n" + "=" * 60)
print("AŞAMA 2: Model Kaydetme & Yükleme")
print("=" * 60)

# Model parametrelerini .nv formatında kaydet ve geri yükle
# (tam model serileştirme için wrapper gerekir)

print("\n" + "=" * 60)
print("AŞAMA 3: INT8 Kuantizasyon")
print("=" * 60)

q_fc1 = nv.QuantizedLinear(16, 32)
q_fc1.load_fp32_weights(fc1.w.data, fc1.bias.data if fc1.bias is not None else None)
q_out = q_fc1(X)
print(f"INT8 çıktı boyutu: {q_out.data.shape}")
```

### C++ Saf Çıkarım (examples/cpp_inference_demo.cpp)

Python bağımlılığı olmadan C++ ile çıkarım. `include/newton/newton.hpp` içindeki `nv::Model` sınıfından türetip katmanları, `forward()` ve `state_dict()` metotlarını tanımlarsınız; ağırlıklar `nv::load_model(model, "model.nv.bin")` ile yüklenir. Tam örnek (LeNet) [`examples/cpp_inference_demo.cpp`](../examples/cpp_inference_demo.cpp) dosyasındadır.

```bash
# Derleme (GCC/MinGW) — başlıklar header-only değildir, cpp/layers.cpp ile linklenir:
g++ -std=c++17 -O2 -fopenmp -Iinclude -Icpp examples/cpp_inference_demo.cpp cpp/layers.cpp -o cpp_inference_demo
```

> Bu derleme `tests/test_cpp_build.py` tarafından otomatik test edilir. Python'dan kaydedilen `.nv.bin` anahtarlarının (ör. `conv1.w`) C++ tarafındaki `state_dict()` anahtarlarıyla (ör. `conv1.weight`) eşleşmesi gerekir.

---

## 🧪 Uygulamalar ve CLI Araçları (`apps/`)

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
