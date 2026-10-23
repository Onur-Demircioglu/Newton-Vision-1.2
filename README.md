# 🌌 newton-vision

> **C++17 ve CUDA çekirdekli, PyTorch ekosistemiyle uyumlu, eğitim ve araştırma odaklı hafif derin öğrenme framework'ü.**

[![Version](https://img.shields.io/badge/version-v1.1.6-blue.svg)](https://github.com/onur-newton/newton-vision)
[![PyTorch Parity](https://img.shields.io/badge/PyTorch%20Parity-85%2F85%20PASS-brightgreen.svg)](benchmarks/VERIFICATION_REPORT.md)
[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-94%2F94%20PASS-brightgreen.svg)](tests/)
[![Math Spec](https://img.shields.io/badge/Math%20Spec-Documented-orange.svg)](MATHEMATICS.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## İçindekiler

1. [Proje Nedir?](#-proje-nedir)
2. [Kurulum](#-kurulum)
3. [Modül Haritası](#-modül-haritası--api-referansı)
4. [Hızlı Başlangıç — 3 Senaryo](#-hızlı-başlangıç--3-temel-senaryo)
5. [Model Zoo — Hazır Mimariler](#-model-zoo--hazır-mimariler)
6. [Katman Referansı — Tüm Layer API](#-katman-referansı--tüm-layer-api)
7. [Tensor & Autograd Motoru](#-tensor--autograd-motoru-fluxion)
8. [Kayıp Fonksiyonları & Optimizatörler](#-kayıp-fonksiyonları--optimizatörler-gravity)
9. [INT8 Kuantizasyon Motoru](#-int8-kuantizasyon-motoru)
10. [JIT & Serileştirme Motoru](#-jit--serileştirme-motoru)
11. [PyTorch Ağırlık Aktarımı](#-pytorch-ağırlık-aktarımı)
12. [Cihaz Yönetimi — CPU & CUDA](#-cihaz-yönetimi--cpu--cuda)
13. [C++ Backend Mimarisi](#-c-backend-mimarisi)
14. [Matematiksel Şartname & Sayısal Doğrulama](#-matematiksel-şartname--sayısal-doğrulama)
15. [Benchmark Sonuçları & Performans](#-benchmark-sonuçları--performans)
16. [Transformer & LLM Yetkinliği](#-transformer--llm-yetkinliği)
17. [Örnek Projeler & Öğreticiler](#-örnek-projeler--öğreticiler)
18. [Test Kapsamı](#-test-kapsamı)
19. [Katkıda Bulunma](#-katkıda-bulunma)
20. [Güvenlik Politikası](#-güvenlik-politikası)
21. [Sürüm Geçmişi](#-sürüm-geçmişi)
22. [Lisans](#-lisans)

---

## ❓ Proje Nedir?

### Bu Framework Ne Yapar?

`newton-vision` aşağıdaki yetenekleri **sıfırdan** C++17, CUDA ve Python ile gerçekleştirir:

| Yetenek | Açıklama |
|---|---|
| **Reverse-Mode Autograd** | Topolojik sıralama ile otomatik türev hesaplama (backward pass) |
| **CNN Katmanları** | Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d (C++ im2col + GEMM ile) |
| **Transformer Katmanları** | MultiHeadAttention, LayerNorm, GELU, PatchEmbedding |
| **Model Zoo** | ResNet-18, MobileNetV2, ViT-Tiny, ViT-Base |
| **Eğitim Döngüsü** | CrossEntropyLoss, MSELoss, Adam, SGD (momentum destekli) |
| **INT8 Kuantizasyon** | Per-Channel ve Global INT8 ağırlık kuantizasyonu (bellek tasarrufu), QuantizedLinear |
| **PyTorch Uyumluluğu** | NumPy'a çevrilmiş PyTorch `state_dict` ağırlıklarını yükleme |
| **JIT / Serileştirme** | `.nv` (sıkıştırılmış), `.nv.bin` (saf C++ ikili), gerçek ONNX (opset 17) dışa aktarımı, onnxruntime ile doğrulanır |
| **InferenceEngine** | FPS ve gecikme takipli canlı çıkarım motoru |
| **Cihaz Yönetimi** | AVX2 SIMD otomatik tespiti; CUDA yalnızca CUDA ile derlenmiş build'lerde (conv2d/relu/maxpool forward) |

### Neden Var?

Derin öğrenme algoritmalarının arkasındaki matematiksel ve donanımsal mantığı (L1/L2 Cache Locality, SIMD AVX2, Topolojik Autograd Sıralaması) **şeffaf** bir şekilde göstermek, deneyler yapmak ve C++ ortamında bağımsız hızlı çıkarım çalıştırmak için geliştirilmiştir.

### Kim Kullanmalı?

- Derin öğrenme motorlarının **iç çalışma mekanizmasını** öğrenmek isteyen araştırmacılar ve öğrenciler
- PyTorch'ta eğittiği modelleri **Python bağımlılığı olmadan** C++ ortamında çalıştırmak isteyen mühendisler
- INT8 kuantizasyon ile **bellek optimizasyonu** yapmak isteyen geliştiriciler

### Kim Kullanmamalı?

- Petabaytlarca veri üzerinde binlerce GPU ile devasa LLM eğitimi yapacak büyük endüstriyel dağıtık sistem ekipleri (PyTorch / JAX kullanmaya devam etmelidirler)

### PyTorch İle İlişkisi Ne?

`newton-vision`, PyTorch'un rakibi veya "katili" **değildir**. PyTorch `state_dict` ağırlıklarını (NumPy dizilerine çevrildikten sonra) aynı mimarideki modellere yükleyebilen, eğitim ve araştırma amaçlı bir altyapıdır. Katman düzeyindeki sayısal parite [`benchmarks/VERIFICATION_REPORT.md`](benchmarks/VERIFICATION_REPORT.md) ile ölçülmüştür.

---

## 🚀 Kurulum

```bash
pip install newton-vision
```

**Gereksinimler:**
- Python 3.9+
- NumPy >= 1.20.0
- C++17 uyumlu derleyici (build için: MSVC, GCC, Clang)
- CUDA Toolkit (opsiyonel, GPU ivmelendirme için)

---

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

## ⚡ Hızlı Başlangıç — 3 Temel Senaryo

### Senaryo 1: Sıfırdan Model Eğitimi (Sadece newton-vision)

Hiçbir harici framework'e ihtiyaç duymadan, saf Autograd türev grafı ile model eğitimi:

```python
import newton_vision as nv
import numpy as np

# ── 1. Sentetik Veri Seti ──────────────────────────────────────────
np.random.seed(42)  # Tekrarlanabilirlik
X_train = np.random.randn(80, 16).astype(np.float32)
y_train = np.random.randint(0, 3, size=(80,)).astype(np.int64)

# ── 2. Model Mimarisi (2 Katmanlı MLP) ────────────────────────────
fc1 = nv.Linear(16, 32)
relu = nv.ReLU()
fc2 = nv.Linear(32, 3)

# ── 3. Optimizatör & Kayıp ────────────────────────────────────────
all_params = fc1.parameters() + fc2.parameters()
optimizer = nv.gravity.Adam(all_params, lr=0.02)
loss_fn = nv.gravity.CrossEntropyLoss()

# ── 4. Eğitim Döngüsü (30 Epoch) ─────────────────────────────────
for epoch in range(1, 31):
    x = nv.Tensor(X_train, requires_grad=True)

    # Forward Pass
    logits = fc2(relu(fc1(x)))
    loss = loss_fn(logits, y_train)

    # Backward Pass (Otomatik Türev)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = np.argmax(logits.data, axis=1)
        acc = np.mean(pred == y_train) * 100
        print(f"Epoch {epoch:02d}/30 | Loss: {float(loss.data):.4f} | Acc: {acc:.1f}%")

# Örnek çıktı (Windows 11, Python 3.13, NumPy 2.x; ağırlık başlatma
# np.random.seed'den sonra yapıldığı için platforma göre küçük farklar olabilir):
# Epoch 10/30 | Loss: 0.8499 | Acc: 56.2%
# Epoch 20/30 | Loss: 0.4879 | Acc: 85.0%
# Epoch 30/30 | Loss: 0.2735 | Acc: 93.8%
```

> **Not:** Etiketler rastgele üretildiği için bu örnek genellenebilir bir öğrenmeyi değil, modelin eğitim verisini **ezberleyebildiğini** gösterir; amacı forward → loss → backward → optimizer zincirinin uçtan uca çalıştığını göstermektir. Gerçek bir öğrenme örneği için `tutorials/01_train_mnist.py`'ye bakın.

**Bu Kod Ne Yapıyor:**
1. 80 örnekli 3 sınıflı sentetik (rastgele etiketli) veri oluşturur
2. `Linear(16→32) → ReLU → Linear(32→3)` mimarisinde bir MLP tanımlar
3. Adam optimizatörü ile CrossEntropy kaybını minimize eder
4. Her epoch'ta `loss.backward()` çağrısı ile topolojik sıralamaya göre tüm parametrelerin gradyanlarını otomatik hesaplar

---

### Senaryo 2: ViT-Tiny ile Çıkarım (Inference)

```python
import newton_vision as nv
import numpy as np

# ── 1. Görüntü Tensörü (Batch=1, C=3, H=224, W=224) ──────────────
x = nv.Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))

# ── 2. Vision Transformer Modeli ──────────────────────────────────
model = nv.models.vit_tiny(num_classes=1000)

# ── 3. İleri Geçiş ───────────────────────────────────────────────
logits = model(x)
print(f"Çıktı boyutu: {logits.data.shape}")  # (1, 1000)

# ── 4. Tahmin ─────────────────────────────────────────────────────
predicted_class = np.argmax(logits.data, axis=1)
print(f"Tahmin edilen sınıf: {predicted_class[0]}")
```

**ViT-Tiny Mimarisi:**
- Embed Dim: 192, Depth: 12 Transformer blok, Heads: 3
- Patch Size: 16×16, Image Size: 224×224
- Parametre sayısı: ~5.7M
- Toplam katman akışı: `PatchEmbedding → [CLS Token + Positional Embedding] → 12x TransformerBlock → LayerNorm → Linear Head`

---

### Senaryo 3: PyTorch Ağırlık Aktarımı & INT8 Kuantizasyon

```python
import newton_vision as nv
import numpy as np

# ── PyTorch'ta eğitilmiş ağırlıkları yükle ────────────────────────
# (PyTorch kurulu değilse bile .npy dosyasından yüklenebilir)
try:
    import torch
    torch_model = torch.nn.Linear(10, 5)
    w_np = torch_model.weight.detach().numpy().T  # PyTorch: (out, in) → newton: (in, out)
    b_np = torch_model.bias.detach().numpy()
except ImportError:
    # PyTorch yoksa rastgele ağırlık
    w_np = np.random.randn(10, 5).astype(np.float32) * 0.1
    b_np = np.zeros(5, dtype=np.float32)

# ── INT8 Kuantize Katman Oluştur ──────────────────────────────────
q_layer = nv.QuantizedLinear(in_features=10, out_features=5)
q_layer.load_fp32_weights(w_np, b_np)

# ── 4x Küçük Bellekle Çıkarım ────────────────────────────────────
test_input = np.random.randn(4, 10).astype(np.float32)
output = q_layer(test_input)
print(f"INT8 çıktı boyutu: {output.data.shape}")  # (4, 5)

# ── Bellek Tasarrufu Hesaplama ────────────────────────────────────
fp32_size = 10 * 5 * 4  # 200 byte
int8_size = 10 * 5 * 1 + 4  # 54 byte (ağırlıklar + scale)
print(f"Bellek tasarrufu: {(1 - int8_size/fp32_size)*100:.1f}%")  # ~73%
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

## 🧊 INT8 Kuantizasyon Motoru

### Neden Kuantizasyon?

FP32 ağırlıkları INT8'e dönüştürerek:
- **~4x ağırlık bellek tasarrufu** (4 byte → 1 byte per parametre, artı ölçek değerleri)
- **Düşük doğruluk kaybı** (Per-Channel kuantizasyon ile)

**Hız hakkında dürüst not:** Mevcut sürümde INT8 kuantizasyonu bir **bellek** optimizasyonudur, hız optimizasyonu değildir. `QuantizedConv2d` her forward'da ağırlıkları FP32'ye geri çevirip FP32 evrişim çalıştırır; `QuantizedLinear` ise INT32 birikimli matris çarpımını NumPy ile yapar. C++ tarafındaki `gemm_int8_cpu` henüz Python'a bağlı değildir. Ölçümlerde INT8 model FP32'den biraz daha yavaştır (bkz. [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md)).

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

### CUDA build'i (Windows, MSVC + CUDA)

```bat
scripts\build_msvc_cuda.bat
```

- **Çıktı:** `build\msvc-cuda\_core.cp313-win_amd64.pyd`. Varsayılan MinGW build'ine dokunmaz.
- **Gereksinimler:** CUDA 12.4 ve Visual Studio'da **MSVC 14.44** araç seti. CUDA 12.4, VS 18'in varsayılan MSVC 14.51'iyle derlerken çöker; script bu yüzden 14.44'ü seçer. Yollar scriptin başında tanımlıdır.
- **Çalışma zamanı:** `cudart64_12.dll` için `CUDA_PATH` tanımlı olmalı. newton-vision onu `os.add_dll_directory` ile ekler.
- **Kapsam:**
  - CUDA'da çalışan işlemler: `conv2d` (stride, padding, dilation, groups), `relu` ve `maxpool2d`; hepsi yalnızca **forward**.
  - Her çağrıda host ↔ device kopyası yapılır. Diğer tüm işlemler ve tüm backward hesapları CPU'dadır.
  - Doğrulama: `tests/test_cuda_kernels.py` çekirdekleri ve ResNet-18 / MobileNetV2'nin tam forward'ını CPU ile karşılaştırır.
  - Bağımsız C++ API'si (`include/newton`) yalnızca CPU'dur.
- **Hız:** GTX 1650'de büyük bir evrişim (8×64×56×56, 3×3) CPU'da 67.6 ms, CUDA'da 19.1 ms sürdü. Bu build'in CPU yolu MinGW build'inden yavaştır, çünkü MSVC'de `omp simd` yok.

**AVX2 SIMD Tespiti:**
C++ backend çalışma zamanında `CPUID` komutu ile CPU'nun AVX2 desteğini kontrol eder. AVX2 varsa SIMD hızlandırılmış GEMM kullanılır; yoksa fallback yoluna geçilir. Bu sayede ARM (Apple Silicon) veya eski x86 işlemcilerde `SIGILL` çökmesi yaşanmaz.

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

Bu yaklaşım evrişimi tek bir büyük matris çarpımına dönüştürerek L1/L2 önbellek yerelliğini (cache locality) iyileştirir ve AVX2 GEMM çekirdeğinin kullanılmasını sağlar. Naif döngüye göre hızlanma bu repoda ayrıca ölçülmemiştir; ayrıntılar için [`research/0001-im2col-vs-direct-conv.md`](research/0001-im2col-vs-direct-conv.md).

---

---

## 📐 Matematiksel Şartname & Sayısal Doğrulama

newton-vision, yalnızca bir kod deposu değil; her katmanın analitik türevlerinin, numerik kararlılık sınırlarının ve matris cebirinin tam olarak belgelendiği bir referans sistemdir:

* 📄 **Tam Matematiksel Şartname:** [`MATHEMATICS.md`](MATHEMATICS.md) — 14 bölüm, simgeler tablosu (nomenclature), gruplu evrişim GEMM formülasyonu, MHSA çok başlıklı zincir kuralı, FP32 Log-Softmax taşma analizi ($88.72$), Taylor 2. derece merkezi fark hata dengesi ($u^{1/3}$ optimum adım tahmini) ve Bessel düzeltmeli BatchNorm.
* 📊 **85-Noktalı Doğrulama Matrisi:** [`benchmarks/VERIFICATION_REPORT.md`](benchmarks/VERIFICATION_REPORT.md) — PyTorch IEEE 754 referansı ile deterministik (`SEED=42`), 8 kategori ve 24 parametrik senaryoda karşılaştırılmış 85 tensör çıktısının hata raporu.

### PyTorch Sayısal Parite Özeti (Örnek Parametreler):

| Kategori | Test Senaryosu | Ölçülen Bileşen | Max Abs Diff | Max Rel Diff | Kosinüs Benzerliği | Durum |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **Adam Optimizatörü** | 10 Adım Yörünge (lr=1e-3, wd=0.05) | Parametre Ağırlıkları | `7.45e-09` | `2.02e-07` | `%100.0000` | **PASS** |
| | | 1. Moment ($m_t$) | `8.94e-08` | `4.59e-06` | `%100.0000` | **PASS** |
| | | 2. Moment ($v_t$) | `1.86e-09` | `1.30e-07` | `%100.0000` | **PASS** |
| **Conv2d** | Dilated + Gruplu ($d=2, G=4$) | Forward ($Y$) | `2.27e-06` | `3.97e-05` | `%100.0000` | **PASS** |
| | | Girdi Gradyanı ($\nabla X$) | `1.97e-06` | `1.27e-04` | `%100.0000` | **PASS** |
| | | Ağırlık Gradyanı ($\nabla W$) | `1.17e-05` | `5.13e-05` | `%100.0000` | **PASS** |
| **Conv2d** | Depthwise Conv ($G=C_{\text{in}}=C_{\text{out}}$) | Ağırlık Gradyanı ($\nabla W$) | `9.14e-06` | `2.54e-06` | `%100.0000` | **PASS** |
| **MHSA (Attention)** | 4 Başlıklı (Head=4, $D=32$) | Forward ($Y$) | `1.32e-06` | `2.48e-05` | `%100.0000` | **PASS** |
| | | Çıkış Projeksiyonu ($\nabla W_O$) | `3.99e-06` | `3.31e-04` | `%100.0000` | **PASS** |
| **CrossEntropy** | Yumuşak Hedefler (Label Smoothing) | Kayıp Değeri | `4.84e-08` | `2.98e-08` | `%100.0000` | **PASS** |
| | | Logits Gradyanı | `2.32e-08` | `3.97e-07` | `%100.0000` | **PASS** |
| **Aktivasyonlar** | GELU (Hendrycks Tanh) | Forward ($Y$) | `3.79e-07` | `1.73e-01` | `%100.0000` | **PASS** |
| | Sigmoid (Uç Noktalar $\pm 500$) | Forward ($Y$) | `1.89e-08` | `4.04e-08` | `%100.0000` | **PASS** |

> *Doğrulama scripti: `python benchmarks/verify_parity_matrix.py` komutuyla bağımsız olarak her an yeniden koşturulabilir.*

---

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

`Intel64 Family 6 (12 Threads)` üzerinde PyTorch v2.6.0 CPU ile karşılaştırma. Ölçüm: `python benchmarks/bench_vs_pytorch.py`, 2026-10-23, `Warmup=3, Runs=10`, `(1, 3, 224, 224)`, iki tarafta da `eval + no_grad`. Script parametre sayıları eşleşmezse durur. Ham rapor: [`benchmarks/BENCH_VS_PYTORCH.md`](benchmarks/BENCH_VS_PYTORCH.md).

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
>
> **Düzeltme:** Önceki sürümlerde ViT-Tiny için "2.28x hızlı" sonucu raporlanmıştı. Bu sonuç geçersizdir: script PyTorch tarafında **ViT-B/16** (86M parametre) ölçüyordu ve newton-vision modelini `eval`/`no_grad` olmadan çalıştırıyordu. İkisi de düzeltildi.

**Yorum:** Tüm model ölçümlerinde PyTorch CPU öndedir. Fark özellikle evrişim yoğun mimarilerde büyüktür; PyTorch'un oneDNN/MKL backend'i yıllardır optimize edilmektedir, depthwise evrişim için özel bir çekirdek hâlâ yoktur (genel gruplu im2col yolu, gruplar üzerinden paralel). newton-vision'ın değeri hızdan çok iç işleyişin şeffaf ve doğrulanabilir olmasıdır.

---

## 🤖 Transformer & LLM Yetkinliği

### Mevcut Durum (v1.1.x)

`newton-vision` aşağıdaki Transformer bileşenlerini **tamamen sıfırdan** implemente etmiştir:

| Bileşen | Durum | Açıklama |
|---|---|---|
| `MultiHeadAttention` | ✅ Tam | Q/K/V projeksiyonları, Scaled Dot-Product Attention, çıkış projeksiyonu |
| `LayerNorm` | ✅ Tam | Öğrenilebilir γ ve β parametreleri |
| `GELU` | ✅ Tam | Backward destekli |
| `TransformerBlock` | ✅ Tam | Pre-LN yapısı, MLP bloğu, Residual Connection |
| `PatchEmbedding` | ✅ Tam | Conv2d tabanlı patch tokenizasyonu |
| `VisionTransformer` | ✅ Tam | CLS token, Positional Embedding, Classification Head |
| `CrossEntropyLoss` | ✅ Tam | Autograd backward destekli |
| `Adam` | ✅ Tam | Bias-corrected moment estimation |

### LLM Eğitebilir mi?

**Kısa cevap: Yapı taşları mevcut, ancak henüz production-ready bir LLM eğitim pipeline'ı yok.**

**Mevcut olan:**
- Transformer encoder blokları (attention + MLP + residual)
- Autograd motoru (backward pass)
- Adam optimizatörü
- CrossEntropyLoss

**Henüz olmayan:**
- Causal/Masked Self-Attention (decoder-only mimariler için)
- Alt-kelime (BPE/WordPiece) tokenizer (basit kelime/karakter tokenizer'ı `nv.text.Tokenizer` ve `nv.Embedding` mevcut)
- Positional Encoding (sinüzoidal veya RoPE)
- Dağıtık eğitim (multi-GPU, data parallelism)
- Gradient checkpointing / mixed precision (FP16/BF16)
- Flash Attention veya benzeri memory-efficient attention
- Büyük veri seti yükleme pipeline'ı

**Sonuç:** `newton-vision` şu an Vision Transformer (ViT) ile **görsel sınıflandırma** eğitimi ve çıkarımı yapabilir. Metin tabanlı LLM (GPT, LLaMA tarzı) eğitimi için yukarıdaki bileşenlerin eklenmesi gerekmektedir. Transformer altyapısı mevcuttur ve bu bileşenler üzerine inşa edilebilir — bu, yol haritamızın bir parçasıdır.

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

Python bağımlılığı olmadan C++ ile çıkarım. `include/newton/newton.hpp` içindeki `nv::Model` sınıfından türetip katmanları, `forward()` ve `state_dict()` metotlarını tanımlarsınız; ağırlıklar `nv::load_model(model, "model.nv.bin")` ile yüklenir. Tam örnek (LeNet) [`examples/cpp_inference_demo.cpp`](examples/cpp_inference_demo.cpp) dosyasındadır.

```bash
# Derleme (GCC/MinGW) — başlıklar header-only değildir, cpp/layers.cpp ile linklenir:
g++ -std=c++17 -O2 -fopenmp -Iinclude -Icpp examples/cpp_inference_demo.cpp cpp/layers.cpp -o cpp_inference_demo
```

> Bu derleme `tests/test_cpp_build.py` tarafından otomatik test edilir. Python'dan kaydedilen `.nv.bin` anahtarlarının (ör. `conv1.w`) C++ tarafındaki `state_dict()` anahtarlarıyla (ör. `conv1.weight`) eşleşmesi gerekir.

---

## ✅ Test Kapsamı

Proje aşağıdaki test ve doğrulama modüllerini içerir (tümü `tests/` ve `benchmarks/` dizininde):

| Test Dosyası | Kapsam | Durum |
|---|---|:---:|
| `tests/test_math_spec.py` | 15 analitik spesifikasyon testi (γ≠1 BN, LN 3D, Soft CE, GELU Tanh) | **PASS** |
| `benchmarks/verify_parity_matrix.py` | 8 Kategori, 24 Parametrik Senaryo, 85 Ayrık Tensör Çıktı Noktası (PyTorch FP64 Paritesi) | **PASS** |
| `tests/test_autograd.py` | Tensor operasyonları, backward pass, topolojik DAG gradyan doğruluğu | **PASS** |
| `tests/test_correctness.py` | Conv2d, Linear, ReLU çıktı doğruluğu (`python_ref/` saf NumPy referansıyla karşılaştırma) | **PASS** |
| `tests/test_models.py` | ResNet-18, MobileNetV2 forward pass boyut doğruluğu | **PASS** |
| `tests/test_vit.py` | VisionTransformer, PatchEmbedding, TransformerBlock doğruluğu | **PASS** |
| `tests/test_quantization.py` | INT8 kuantizasyon, dequantizasyon, QuantizedLinear doğruluğu | **PASS** |
| `tests/test_jit.py` | Model kaydetme/yükleme, .nv ve .nv.bin formatları | **PASS** |
| `tests/test_real_training.py` | Lineer regresyon ve XOR uçtan uca eğitim (gerçek öğrenme doğrulaması) | **PASS** |
| `tests/test_training_components.py` | Flatten, Conv1d, BatchNorm1d, Embedding, Upsample (nearest/linear/bilinear), L1/BCE/BCEWithLogits kayıpları, AdamW, `clip_grad_norm_`, StepLR/CosineAnnealingLR, optimizer `state_dict` ile devam, veri artırma, tokenizer, 5000 düğümlük graf backward'ı. Hepsi PyTorch (float64) ile karşılaştırılır. Ayrıca iki uçtan uca eğitim testi var; ikisi de görülmemiş veride %95'in üzerine çıkmalı | **PASS** |
| `tests/test_onnx_export.py` | ResNet-18, MobileNetV2, ViT ve yeni katmanlar gerçek ONNX'e aktarılıp onnxruntime ile karşılaştırılır. Grafı kopuk modelin ve desteklenmeyen katmanın reddedildiği de test edilir. onnx yoksa SKIP | **PASS** |
| `tests/test_cuda_kernels.py` | CUDA conv2d (dilation/groups/depthwise, >65535 çıktı kanalı×batch), relu ve maxpool2d çekirdekleri; `.cuda()` tensörlerle katman forward/gradyanları; ResNet-18 ve MobileNetV2 tam forward. Hepsi CPU ile karşılaştırılır. CUDA build'i yoksa SKIP | **PASS** (MSVC + CUDA build'i) |
| `tests/test_cpp_build.py` | `tests/test_cpp_core.cpp` (C++ çekirdek self-test) ve `examples/cpp_inference_demo.cpp` derlenip çalıştırılır. g++ yoksa açıkça SKIP edilir | **PASS** |

> **Not:** Modül düzeyinde `test_*` fonksiyonları kullanan dosyalar (`test_autograd`, `test_correctness`, `test_jit`, `test_models`, `test_real_training`) unittest'in `load_tests` protokolüyle toplanır. Her dosya `python tests/<dosya>.py` ile tek başına da çalıştırılabilir.

Testleri çalıştırma:

```bash
# Tüm testler (94 test; CUDA build'i yoksa 5 CUDA testi SKIP):
python -m unittest discover tests

# 85-Noktalı Resmi PyTorch Doğrulama Matrisi:
python benchmarks/verify_parity_matrix.py

# Tek tıkla resmi benchmark suiti:
python benchmarks/run_all.py
```

---

## 🤝 Katkıda Bulunma

1. Depoyu fork'layın
2. Feature branch oluşturun: `git checkout -b feature/yeni-ozellik`
3. Değişikliklerinizi commit'leyin: `git commit -m 'feat: yeni özellik ekle'`
4. Branch'inizi push'layın: `git push origin feature/yeni-ozellik`
5. Pull Request açın

**Commit Mesajı Formatı:** Conventional Commits (feat:, fix:, docs:, refactor:, test:, perf:)

**Kod Stili:** PEP 8 (Python), Google C++ Style Guide (C++)

---

## 🔒 Güvenlik Politikası

Güvenlik açığı bulursanız:
- **Public issue açmayın**
- E-posta ile bildirin: `captainonur86@gmail.com`
- 48 saat içinde yanıt alacaksınız

Desteklenen sürümler: v1.1.x (en güncel minor sürüm)

---

## 📋 Sürüm Geçmişi

### v1.1.6 (Güncel - Numerically Verified)
- **Matematiksel Şartname ([`MATHEMATICS.md`](MATHEMATICS.md)):** 14 bölüm, simgeler tablosu, gruplu evrişim GEMM formülasyonu ve MHSA analitik türev zincirleri eksiksiz belgelendi.
- **85-Noktalı Doğrulama Matrisi ([`benchmarks/VERIFICATION_REPORT.md`](benchmarks/VERIFICATION_REPORT.md)):** Deterministik seed (`SEED=42`) ile PyTorch IEEE 754 referansına karşı 85 tensör noktası doğrulandı (%100 PASS).
- **CrossEntropy Kararlılığı:** `np.log(p + 1e-12)` hilesi kaldırıldı; sayısal kararlı Log-Softmax formülasyonuna geçilerek kayıp ve gradyan analitik olarak tam eşitlendi.
- **GELU Analitik Türevi:** Hendrycks Tanh yaklaşımının türevi düzeltildi; PyTorch `approximate="tanh"` ile tam sayısal parite sağlandı.
- **BatchNorm2d Bessel Düzeltmesi:** `running_var` için $M/(M-1)$ çarpanı eklendi; tekil örneklem ($M \le 1$) guard'ı ve eval-mode backward yolu tanımlandı.
- **INT8 Simetrik Sınırı:** $[-127, 127]$ sınırları sıkılaştırıldı; yuvarlama NumPy `np.round` (ties-to-even, Banker's rounding) ile yapılır.

### v1.1.4
- README tamamen yeniden yazıldı (PyPI uyumlu, self-contained)
- Tüm dahili link bağımlılıkları kaldırıldı
- LLM yetkinlik durumu belgelendi

### v1.1.0
- Vision Transformer (ViT-Tiny, ViT-Base) eklendi
- MultiHeadAttention, LayerNorm, GELU katmanları eklendi
- PatchEmbedding ve TransformerBlock eklendi
- Per-Channel INT8 Kuantizasyon eklendi
- InferenceEngine (FPS/gecikme takibi) eklendi
- ONNX graf tanımlayıcısı eklendi (o sürümde gerçek ONNX değildi; gerçek ONNX dışa aktarımı v1.1.6 sonrası eklendi)

### v1.0.0
- İlk kararlı sürüm
- Conv2d (im2col + GEMM), Linear, ReLU, MaxPool2d, AvgPool2d, BatchNorm2d
- Reverse-mode Autograd motoru
- ResNet-18 ve MobileNetV2
- PyTorch state_dict uyumluluğu
- .nv ve .nv.bin serileştirme
- CPU/CUDA cihaz yönetimi
- AVX2 SIMD otomatik tespiti

---

## 📜 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır. Telif hakkı (c) 2025 newton-vision Core Team.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

---

> **newton-vision** — Derin öğrenmenin fiziğini anla, motorunu kendin yaz. 🌌
