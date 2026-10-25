# Testler ve Sayısal Doğrulama

Testleri çalıştırma, kapsam ve PyTorch'a karşı sayısal parite.

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

## 📐 Matematiksel Şartname & Sayısal Doğrulama

newton-vision, yalnızca bir kod deposu değil; her katmanın analitik türevlerinin, numerik kararlılık sınırlarının ve matris cebirinin tam olarak belgelendiği bir referans sistemdir:

* 📄 **Tam Matematiksel Şartname:** [`docs/mathematics.md`](mathematics.md) — 14 bölüm, simgeler tablosu (nomenclature), gruplu evrişim GEMM formülasyonu, MHSA çok başlıklı zincir kuralı, FP32 Log-Softmax taşma analizi ($88.72$), Taylor 2. derece merkezi fark hata dengesi ($u^{1/3}$ optimum adım tahmini) ve Bessel düzeltmeli BatchNorm.
* 📊 **85-Noktalı Doğrulama Matrisi:** [`benchmarks/VERIFICATION_REPORT.md`](../benchmarks/VERIFICATION_REPORT.md) — PyTorch IEEE 754 referansı ile deterministik (`SEED=42`), 8 kategori ve 24 parametrik senaryoda karşılaştırılmış 85 tensör çıktısının hata raporu.

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
