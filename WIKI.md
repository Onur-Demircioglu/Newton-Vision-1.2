# 📚 newton-vision GitHub Wiki & Mimari Kullanım Rehberi

Welcome to the **newton-vision** official GitHub Wiki!

`newton-vision` is a lightweight, educational deep learning framework with a C++17 core (AVX2 + OpenMP, optional CUDA forward kernels), a NumPy-based reverse-mode Autograd, PyTorch state_dict weight loading, and INT8 weight quantization for memory savings.

---

## 📌 İçindekiler / Table of Contents
1. [Kurulum ve Derleme (Installation & Compilation)](#1-kurulum-ve-derleme)
2. [Hızlı Başlangıç (Quick Start Guide)](#2-hızlı-başlangıç)
3. [Modül Hiyerarşisi (Module Architecture)](#3-modül-hiyerarşisi)
4. [Saf C++ İkili Motoru (Pure C++ Standalone Engine)](#4-saf-c-ikili-motoru)
5. [INT8 Kuantizasyon (INT8 Quantization)](#5-int8-kuantizasyon)
6. [Performans Benchmarkları (Performance Benchmarks)](#6-performans-benchmarkları)

---

## 1. Kurulum ve Derleme

### PyPI Üzerinden Yükleme:
```bash
pip install newton-vision
```

### Kaynak Koddan Derleme (C++ & CUDA):
```bash
git clone https://github.com/onur/newton-vision.git
cd newton-vision
pip install . --no-build-isolation
```

---

## 2. Hızlı Başlangıç

```python
import newton_vision as nv
import numpy as np

# 1. Tensör Oluşturma
x = nv.Tensor(np.random.randn(1, 3, 224, 224), requires_grad=True)

# 2. Vision Transformer Modeli
model = nv.models.vit_tiny(num_classes=1000)

# 3. İleri Geçiş (Forward Pass)
logits = model(x)
print("Logits Shape:", logits.shape)
```

---

## 3. Modül Hiyerarşisi

- `nv.optics`: Vision Layers (`Conv2d`, `Linear`, `BatchNorm2d`, `LayerNorm`, `GELU`, `MultiHeadAttention`)
- `nv.fluxion`: Reverse-mode Autograd Engine (`Tensor`, `.backward()`, `.cuda()`, `.cpu()`)
- `nv.gravity`: Optimizers & Losses (`Adam`, `SGD`, `MSELoss`, `CrossEntropyLoss`)
- `nv.models`: Vision Model Zoo (`resnet18`, `mobilenet_v2`, `vit_tiny`, `vit_base`)
- `nv.quantization`: INT8 Quantization (`quantize_model`, `quantize_tensor_per_channel`)
- `nv.jit`: Serialization & Server (`save_model_bin`, `load_model_bin`, `InferenceEngine`)

---

## 4. Saf C++ İkili Motoru

Python olmadan C++17 ortamında model çalıştırmak için `#include <newton/newton.hpp>` başlığını projenize dahil edin. `nv::Model` soyut bir sınıftır: katmanları ve `forward()` / `state_dict()` metotlarını kendiniz tanımlarsınız, ağırlıkları `nv::load_model()` ile `.nv.bin` dosyasından yüklersiniz. Tam ve çalışan örnek: [`examples/cpp_inference_demo.cpp`](examples/cpp_inference_demo.cpp).

```cpp
#include "newton/newton.hpp"

class TinyNet : public nv::Model {
public:
    nv::Linear fc{784, 10};
    std::map<std::string, nv::Tensor*> params{{"fc.weight", &fc.weight}, {"fc.bias", &fc.bias}};

    std::map<std::string, nv::Tensor*>& state_dict() override { return params; }
    nv::Tensor forward(const nv::Tensor& x) override { return fc(x); }
};

int main() {
    TinyNet model;
    nv::load_model(model, "model.nv.bin");  // anahtar adları state_dict() ile eşleşmeli

    nv::Tensor input(nv::Shape{1, 784, 1, 1}, std::vector<float>(784, 1.0f), nv::Device::CPU);
    nv::Tensor output = model(input);
    return 0;
}
```

Derleme (GCC/MinGW; başlık dosyaları header-only değildir, `cpp/layers.cpp` ile linklenmelidir):

```bash
g++ -std=c++17 -O2 -fopenmp -Iinclude -Icpp my_app.cpp cpp/layers.cpp -o my_app
```

---

## 5. INT8 Kuantizasyon

Ağırlık belleğini ~%75 azaltır; hesaplama INT8 SIMD ile yapılmaz, bu yüzden hız kazancı yoktur.

```python
import newton_vision as nv

model = nv.models.resnet18()
stats = nv.quantize_model(model, per_channel=True)
print(f"Memory Reduction: %{stats['memory_reduction_percent']:.2f}")
```

---

## 6. Performans Benchmarkları

| Model | Batch Size | newton-vision | PyTorch 2.6 CPU | Hız Farkı |
|---|---|---|---|---|
| ViT-Tiny | 1 | 61.25 ms | 24.18 ms (aynı boyutlu ViT-Tiny) | 2.53x Yavaş |
| ResNet-18 | 1 | 122.97 ms | 29.34 ms | 4.19x Yavaş |
| MobileNetV2 | 1 | 109.18 ms | 25.18 ms | 4.34x Yavaş |

> 2026-10-23, `python benchmarks/bench_vs_pytorch.py` (`eval + no_grad`, parametre sayıları eşit). Mutlak süreler koşudan koşuya değişkendir. Önceki "3.22x Hızlı" ViT sonucu, PyTorch tarafında ViT-B/16 (86M parametre) ölçüldüğü için geçersizdir. Ayrıntılar: [`benchmarks/BENCH_VS_PYTORCH.md`](benchmarks/BENCH_VS_PYTORCH.md).
