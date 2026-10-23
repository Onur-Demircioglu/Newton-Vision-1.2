# 📜 CHANGELOG — Sürüm Geçmişi

`newton-vision` projesindeki tüm önemli değişiklikler bu dosyada Semantik Sürümleme (Semantic Versioning) standartlarına uygun olarak belgelenir.

---

## [Yayınlanmamış] - 2026-10-23

### 📝 Dokümantasyon Düzeltmeleri
- README, PROGRESS, AI_MEMORY, ROADMAP, CAPABILITIES ve `docs/` kodla karşılaştırılarak düzeltildi:
  - ViT-Tiny "2.28x hızlı" iddiası kaldırıldı. Benchmark PyTorch tarafında ViT-B/16 kullanıyordu; aynı mimaride ölçülen sonuç ~3x yavaş.
  - CUDA, ONNX, INT8 hızı, malware doğruluğu ve "peer-reviewed" ifadeleri gerçek davranışa göre düzeltildi.
  - README Senaryo 1'deki uydurma "Beklenen Çıktı" gerçek çıktı ile değiştirildi.
  - Sürüm (1.1.6) ve test sayısı (unittest: 39) tüm belgelerde eşitlendi.
### 🔧 Düzeltildi
- `AvgPool2d.forward` çıktı tensörünü döndürmüyordu (`None`); bu yüzden ResNet-18 forward çöküyordu. `return out` eklendi.
- `load_state_dict` yeniden yazıldı; `Module.load_state_dict` artık aynı implementasyonu kullanıyor.
  - Eski hali `gamma`, `beta`, `cls_token`, `pos_embed` gibi adları sessizce atlıyordu. Bu yüzden `save_model → load_model` ViT'te parametrelerin yarısını, ResNet'te BN parametrelerini kaybediyordu. Ayrıca kare Linear ağırlıklarını yanlışlıkla transpoze ediyordu.
  - Şekil uyuşmazlığı artık `ValueError` veriyor ve model değişmeden kalıyor.
  - `strict` uygulanıyor; `LoadResult(missing_keys, unexpected_keys)` dönüyor; eksik/beklenmeyen anahtarlar uyarı veriyor.
  - `load_model` / `load_model_bin` `strict=True` kullanıyor.
- `.nv.bin` round-trip'i Linear katmanını bozuyordu (ağırlık `(in, out, 1, 1)` olarak geri geliyordu); yükleyici bu biçimleri artık geri açıyor.
- torchvision ResNet-18 aktarımı: `downsample.0/1` eşlemesi eklendi. `MaxPool2d(padding=...)` eklendi ve ResNet stem'i `padding=1` oldu. Çıktı farkı 6e-6, kosinüs 1.0.
- `include/newton/tensor.hpp`: eksik `#include <cstdint>` eklendi (GCC 15 derleme hatası).
- `tests/test_cpp_core.cpp` güncel `conv2d` imzasına uyarlandı.
- `benchmarks/bench_vs_pytorch.py`:
  - ViT-B/16 yerine aynı boyutlu ViT-Tiny ile karşılaştırma yapılıyor.
  - newton-vision tarafı `eval + no_grad` ile ölçülüyor.
  - Parametre sayısı kontrolü eklendi.
  - Doğru Warmup/Runs etiketleri koddan geliyor.
  - Rapor artık `BENCHMARK_RESULTS.md`'nin üzerine yazmıyor, `BENCH_VS_PYTORCH.md`'ye yazılıyor.

### 🧪 Testler
- Modül düzeyinde fonksiyon kullanan 5 test dosyası `load_tests` ile `unittest discover`'a bağlandı. `verify_real_training.py` dosyası `test_real_training.py` olarak yeniden adlandırıldı.
- Yeni `tests/test_cpp_build.py`: C++ self-test'i ve C++ çıkarım demosunu derleyip çalıştırır.
- `python -m unittest discover tests`: 57/57.

- **Eğitim tutorial'ları grafı koparıyordu:** `nv.Tensor(x.data.reshape(...))` yüzünden konvolüsyon katmanları eğitilmiyordu (tam MNIST'te ağırlık değişimi 0). Düzeltilen dosyalar: `tutorials/train_and_eval_lenet.py`, `tutorials/mnist_training.py`, `tutorials/01_train_mnist.py`, `examples/01_image_classification_lenet.py`, `apps/image_predictor.py`.
- `tutorials/01_train_mnist.py` rastgele veri yerine gerçek MNIST kullanıyor; INT8 deneyi `nv.quantize_model` ile yapılıyor.
- `tutorials/train_and_eval_lenet.py` artık `lenet_mnist_v1.nv`'nin üzerine yazmıyor; lr 0.008 → 0.001.
- `Tensor.backward`: özyinelemeli topolojik sıralama iteratif yapıldı (uzun graflarda `RecursionError` veriyordu).
- `Tensor.to()`: gradyanı kaynak tensöre aktarmıyordu; türevlenebilir kimlik işlemi yapıldı.

### ✨ Eklendi
- `MaxPool2d(kernel_size, stride=None, padding=0)`: PyTorch ile aynı `-inf` dolgu semantiği.
- `newton_vision.weights.LoadResult`.
- Katmanlar: `Flatten`, `Conv1d`, `BatchNorm1d`, `Embedding`, `Upsample` (nearest).
- Kayıplar: `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`.
- Eğitim: `clip_grad_norm_`, `StepLR`, `CosineAnnealingLR`, `SGD`/`Adam` için `state_dict()`/`load_state_dict()`.
- `nv.transforms` (`Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`) ve `TensorDataset(..., transform=...)`.
- `tests/test_training_components.py`: yeni bileşenlerin PyTorch float64 karşılaştırmaları ve iki uçtan uca eğitim testi.
- **Gerçek ONNX dışa aktarımı** (`nv.export_onnx`, opset 17, `newton_vision/onnx_export.py`):
  - İzleme tabanlı; model zoo'nun tamamı dışa aktarılabiliyor.
  - onnxruntime kuruluysa çıktı iki farklı girdiyle doğrulanıyor.
  - Opsiyonel bağımlılık: `pip install newton-vision[onnx]`.
  - Eski metin tanımlayıcı formatın yerini aldı.
- `AdamW` (ayrık ağırlık azaltma), `Upsample(mode="linear" | "bilinear", align_corners=...)`, `nv.text.Tokenizer` (kelime/karakter).
- `tests/test_onnx_export.py`. Toplam: 89/89.

### ⚡ Performans
- C++ `conv2d_backward`:
  - Seri ve naif döngüler, yeniden sıralanmış ve paralel (OpenMP + `omp simd`) döngülerle değiştirildi.
  - `col2im` kanal bazında paralel çalışıyor.
- Gruplu/depthwise `conv2d` forward ve backward: paralellik gruplar üzerinden. GEMM ve `im2col`'daki iç içe paralel bölgeler `if(!omp_in_parallel())` ile engellendi.
- Eğitim adımı (batch 2, 224², bu makine): ResNet-18 7.36 → 1.16 sn, MobileNetV2 5.26 → 1.03 sn. Parite 85/85 korundu.

### 🛠 Altyapı
- Proje klasöründe ayrı git deposu başlatıldı (`main`). `.gitignore` uygulamaların ihtiyaç duyduğu `models/*.nv`, `tutorials/lenet_mnist_v1.nv` ve `assets/` dosyalarını artık hariç tutmuyor.

---

## [1.1.6] - 2026-10-09

### 🔧 Düzeltildi / Eklendi
- `MATHEMATICS.md` matematiksel şartnamesi ve 85 noktalı PyTorch parite matrisi (`benchmarks/verify_parity_matrix.py`).
- CrossEntropy: `log(p + eps)` yerine sayısal kararlı log-softmax.
- GELU tanh yaklaşımının türevi düzeltildi.
- BatchNorm2d `running_var` için Bessel düzeltmesi.
- INT8 simetrik aralık `[-127, 127]`.

---

## [1.1.2] – [1.1.5] - 2026-07-23

PyPI'a yüklendi; ayrı sürüm notu tutulmadı.

---

## [1.1.0] - 2026-07-23 (Maintainer & Quality Release — PyPI'a yüklenmedi)

### 🚀 Eklendi
- **Bilimsel Benchmark Paketi (`benchmarks/bench_vs_pytorch.py`):** `mean`, `std`, `min`, `max` istatistiksel ölçüm ve PyTorch v2.6.0 CPU karşılaştırma raporu.
- **Akademik Dokümantasyon Seti (`docs/`):** `architecture.md`, `autograd.md`, `performance.md`, `quantization.md`, `benchmark.md`.
- **Zengin Örnek Projeler (`examples/`):** 5 adet çalışabilir örnek proje scripti.
- **Dinamik CPUID Tespiti:** C++ çekirdeğinde `nv::has_avx2_hardware()` kontrolü ile eski CPU ve ARM çiplerinde çökme engellendi.
- **Kanal Bazlı (Per-Channel) INT8 Kuantizasyon:** `quantize_tensor_per_channel` ile filtre bazlı bağımsız ölçekleme.
- **Açık Kaynak Yönetişim Belgeleri:** `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`.

---

## [1.0.0] - 2026-07-19 (İlk Kararlı Sürüm)

### 🚀 Eklendi
- PyPI Paket Yayını (`pip install newton-vision`).
- C++17 OpenMP `im2col + GEMM` evrişim çekirdeği (opsiyonel CUDA forward kernel'ları; yalnızca CUDA ile derlenen build'lerde).
- Ters mod Otomatik Türev motoru (`newton_vision.fluxion`).
- ResNet-18, MobileNetV2 ve LeNet model uygulamaları.
- PyTorch `state_dict` ağırlık aktarıcısı.
