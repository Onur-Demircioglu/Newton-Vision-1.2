# 🌌 newton-vision

> **C++17 çekirdekli, NumPy tabanlı autograd'a sahip, PyTorch benzeri API sunan eğitim ve araştırma odaklı derin öğrenme framework'ü.**

[![Version](https://img.shields.io/badge/version-v1.2.0-blue.svg)](CHANGELOG.md)
[![Tests](https://img.shields.io/badge/Tests-94%2F94%20PASS-brightgreen.svg)](docs/testing.md)
[![PyTorch Parity](https://img.shields.io/badge/PyTorch%20Parity-85%2F85%20PASS-brightgreen.svg)](benchmarks/VERIFICATION_REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

newton-vision, derin öğrenme motorlarının iç işleyişini **şeffaf ve doğrulanabilir** biçimde göstermek için sıfırdan yazıldı. Ağır işlemler (evrişim, havuzlama, normalizasyon) C++ (AVX2 + OpenMP) ile çalışır. Autograd grafı ve API Python'dadır. Her katman PyTorch'a karşı sayısal olarak test edilir.

PyTorch'un rakibi değildir: PyTorch'tan daha yavaştır (bkz. [Sınırlar](#-sınırlar)). Değeri, okunabilir ve ölçülmüş olmasındadır.

---

## ✨ Neler Var?

| Alan | İçerik |
|---|---|
| **Autograd** | Reverse-mode, iteratif topolojik sıralama, `no_grad`, gradcheck testli |
| **Katmanlar** | Conv1d/2d (stride, padding, dilation, groups), Linear, BatchNorm1d/2d, LayerNorm, MultiHeadAttention, Max/AvgPool, Upsample (nearest/linear/bilinear), Embedding, Flatten, Dropout, aktivasyonlar |
| **Modeller** | ResNet-18, MobileNetV2, ViT-Tiny/Base |
| **Eğitim** | CrossEntropy, MSE, L1, BCE, BCEWithLogits; SGD, Adam, AdamW; StepLR, CosineAnnealingLR; `clip_grad_norm_`; optimizer `state_dict` ile eğitime devam |
| **Veri** | `TensorDataset`, `DataLoader`, `ImageFolder`, `CSVDataset`, `nv.transforms`, `nv.text.Tokenizer` |
| **Dışa aktarma** | `.nv` / `.nv.bin`, PyTorch `state_dict` yükleme, gerçek **ONNX** (opset 17, onnxruntime ile doğrulanır) |
| **Kuantizasyon** | Per-channel INT8 ağırlık kuantizasyonu (~%75 ağırlık belleği tasarrufu) |
| **C++ / CUDA** | Bağımsız C++ çıkarım API'si (CPU); opsiyonel CUDA build'i (conv2d / relu / maxpool forward) |

---

## 🚀 Kurulum

```bash
pip install newton-vision            # temel (yalnızca numpy bağımlılığı)
pip install newton-vision[onnx]      # + ONNX dışa aktarımı ve doğrulaması
```

Kaynaktan (C++17 derleyici gerekir): `pip install -e . --no-build-isolation`. CUDA build'i için: [docs/cuda.md](docs/cuda.md).

---

## ⚡ Hızlı Başlangıç

```python
import numpy as np
import newton_vision as nv

model = nv.Sequential(
    nv.Conv2d(1, 8, kernel_size=3, padding=1), nv.ReLU(), nv.MaxPool2d(2),
    nv.Flatten(),
    nv.Linear(8 * 14 * 14, 10),
)
optimizer = nv.Adam(model.parameters(), lr=1e-3)
loss_fn = nv.CrossEntropyLoss()
loader = nv.DataLoader(nv.TensorDataset(x_train, y_train), batch_size=64, shuffle=True)  # x: (N, 1, 28, 28)

for epoch in range(3):
    model.train()
    for bx, by in loader:
        loss = loss_fn(model(bx), by)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
nv.save_model(model, "model.nv")                       # kaydet / nv.load_model ile yükle
nv.export_onnx(model, x_train[:1], "model.onnx")       # gerçek ONNX, onnxruntime ile doğrulanır
```

PyTorch ağırlıklarını yüklemek:

```python
state = {k: v.detach().numpy() for k, v in torchvision.models.resnet18().state_dict().items()}
result = nv.load_state_dict(nv.models.resnet18(), state)   # result.missing_keys == [] beklenir
```

Gerçek veriyle çalışan örnekler: `tutorials/01_train_mnist.py`, `tutorials/train_and_eval_lenet.py`, `tutorials/02_mnist_cnn_vs_pytorch.py` ve `tutorials/03_cifar10_cnn_vs_pytorch.py` (aynı koşullarda PyTorch ile karşılaştırma; CIFAR-10 gerekirse indirilip MD5 ile doğrulanır). Tüm API: [docs/guide.md](docs/guide.md).

---

## 📊 Doğrulanmış Sonuçlar

Hepsi bu repodaki scriptlerle ölçüldü (2026-10-23/25). Ayrıntılar: [docs/performance.md](docs/performance.md), [docs/testing.md](docs/testing.md).

| Ölçüm | Sonuç |
|---|---|
| CIFAR-10, BatchNorm'lu CNN (806K parametre), 15 epoch | **%86.44** test doğruluğu; aynı koşullarda PyTorch %86.25 (eğitim ~6.6x hızlı) |
| Gerçek MNIST, BatchNorm'lu CNN, 10 epoch | **%99.42** test doğruluğu; aynı koşullarda PyTorch %99.47 (eğitim ~8.6x hızlı) |
| Gerçek MNIST, LeNet, 3 epoch | %98.3 test doğruluğu |
| PyTorch 2.6 katman paritesi | 85/85 nokta tolerans içinde (en kötü FP32 farkı 2.7e-5) |
| torchvision ResNet-18 ağırlık aktarımı | çıktı farkı 6e-6, kosinüs 1.0 |
| ResNet-18 eğitim adımı | loss PyTorch ile birebir aynı |
| ONNX dışa aktarımı | ResNet-18, MobileNetV2, ViT onnxruntime ile doğrulandı |
| Eğitim adımı (batch 2, 224², CPU) | ResNet-18 0.88 sn, MobileNetV2 0.68 sn |
| Testler | 94 test (CUDA build'i yoksa 5 CUDA testi SKIP) |

---

## ⚠️ Sınırlar

- **Hız:** PyTorch CPU'dan yavaştır: çıkarımda ResNet-18 ~4.2x, MobileNetV2 ~4.3x, ViT-Tiny ~2.5–3.7x; uçtan uca eğitimde ~6.6x (CIFAR-10 CNN) ve ~8.6x (MNIST CNN).
- **GPU:** GPU ile eğitim yok. CUDA build'inde yalnızca `conv2d`, `relu` ve `maxpool2d` forward'u GPU'da çalışır; her çağrıda host↔device kopyası yapılır, backward CPU'dadır.
- **INT8:** Yalnızca bellek optimizasyonudur, hız kazandırmaz.
- **ONNX:** Graf, örnek girdinin şekline sabittir (dinamik batch yok). Kuantize katmanlar dışa aktarılamaz.
- **LLM:** Causal attention maskesi ve alt-kelime (BPE) tokenizer yoktur; dağıtık eğitim ve mixed precision da yoktur.
- **Bağımsız C++ API'si:** Yalnızca CPU'dur.

Açık işler ve yol haritası: [PROGRESS.md](PROGRESS.md).

---

## 📚 Dokümanlar

| Belge | İçerik |
|---|---|
| [docs/guide.md](docs/guide.md) | Tüm API: tensör/autograd, katmanlar, modeller, eğitim, veri, INT8, serileştirme/ONNX, PyTorch aktarımı, C++ API, örnekler |
| [docs/performance.md](docs/performance.md) | Benchmark'lar ve C++ optimizasyon ayrıntıları |
| [docs/testing.md](docs/testing.md) | Test kapsamı ve PyTorch paritesi |
| [docs/cuda.md](docs/cuda.md) | Windows MSVC + CUDA build'i |
| [docs/architecture.md](docs/architecture.md) | Mimari ve bellek/cihaz modeli |
| [docs/mathematics.md](docs/mathematics.md) | Tüm katmanların matematiksel şartnamesi ve türevleri |
| [docs/quantization.md](docs/quantization.md) | INT8 kuantizasyon matematiği |
| [PROGRESS.md](PROGRESS.md) | Güncel durum, açık sorunlar, yol haritası |
| [CHANGELOG.md](CHANGELOG.md) | Sürüm geçmişi |

---

## 🗂️ Proje Yapısı

```
src/newton_vision/   Python paketi (tensor, layers, gravity, models, onnx_export, ...)
cpp/                 C++ çekirdek (im2col + AVX2 GEMM, OpenMP) ve CUDA çekirdekleri
include/newton/      Bağımsız C++ çıkarım API'si
tests/               unittest testleri (PyTorch / onnxruntime karşılaştırmaları dahil)
examples/            Kısa API örnekleri (+ C++ demo)
tutorials/           Gerçek MNIST ve CIFAR-10 eğitimi (+ PyTorch ile aynı koşullarda karşılaştırma)
apps/                Demo uygulamaları (görüntü tahmini, PyTorch/INT8 karşılaştırması, sentetik malware demosu)
benchmarks/          Parite ve hız ölçümleri (python benchmarks/run_all.py)
scripts/             Derleme scriptleri (MSVC + CUDA)
```

---

## ✅ Testler

```bash
python -m unittest discover tests      # 94 test
python benchmarks/run_all.py           # parite + uçtan uca ölçümler (--full: PyTorch karşılaştırması)
```

---

[Katkıda bulunma](CONTRIBUTING.md) · [Güvenlik](SECURITY.md) · [Lisans (MIT)](LICENSE)
