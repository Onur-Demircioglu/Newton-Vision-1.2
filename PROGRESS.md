# newton-vision — İlerleme & Gelecek Yol Haritası (Single Source of Truth)

> Bu dosya projenin **tek doğruluk kaynağıdır (Single Source of Truth)**.  
> Derinlemesine AI ve mimari hafızası için bkz: [`AI_MEMORY.md`](file:///c:/Users/onur_/Desktop/newton-vision/AI_MEMORY.md).

---

## 📌 Proje Genel Bilgileri

- **Proje Adı:** `newton-vision`
- **Sürüm:** `1.1.6` (`pyproject.toml` ve `__version__` ile aynı; PyPI'daki son sürüm)
- **PyPI Sayfası:** [https://pypi.org/project/newton-vision/](https://pypi.org/project/newton-vision/)
- **Tanım:** C++17 çekirdekli (AVX2 + OpenMP), NumPy tabanlı autograd motoruna sahip, eğitim/araştırma odaklı derin öğrenme kütüphanesi. PyTorch `state_dict` ağırlıklarını (NumPy'a çevrilmiş) yükleyebilir; INT8 ağırlık kuantizasyonu ile bellek tasarrufu sağlar. CUDA kernel'ları (conv2d/relu/maxpool forward) yalnızca CUDA ile derlenen build'lerde etkindir.
- **Konum:** `C:\Users\onur_\Desktop\newton-vision`

---

## 🏆 Tamamlanan Aşamalar & Özellikler (v1.1.6)

### 1. Model Zoo & Modern Mimariler
- **LeNet-5:** Klasik evrişimli ağ ve MNIST sınıflandırma (`tutorials/` altında; `nv.models` içinde `lenet5()` fonksiyonu yoktur).
- **ResNet-18:** 8 Residual Blok, skip connection. Forward ve backward çalışıyor. torchvision ResNet-18 ağırlıklarıyla çıktı farkı en fazla 6e-6, kosinüs 1.0 (2026-10-23). Stem `MaxPool2d(3, 2, padding=1)` artık torchvision ile aynı.
- **MobileNetV2:** Inverted Residuals + Linear Bottlenecks mimarisi ve `ModuleList` parametre takibi.
- **Vision Transformer (ViT-Tiny):** Patch Embedding, Multi-Head Self-Attention (MHSA), GELU aktivasyonu ve LayerNorm.

### 2. Donanım İvmelendirme & INT8 Kuantizasyon
- **Per-Channel INT8 Quantization:** ResNet-18 ağırlıkları 43.64 MB'tan 10.92 MB'a (**%74.96 ağırlık belleği tasarrufu**). Hız kazancı yoktur: INT8 model FP32'den biraz yavaştır (`QuantizedConv2d` ağırlıkları FP32'ye geri çevirir, C++ `gemm_int8_cpu` Python'a bağlı değildir).
- **AVX2 SIMD:** 4 katmanlı küçük CNN, `(1, 3, 32, 32)`, batch=1: **~540 – 780 FPS** (ölçüme göre değişir). Büyük modellerde PyTorch CPU belirgin şekilde daha hızlıdır: ResNet-18 ~4.2x, MobileNetV2 ~4.3x (C++ optimizasyonundan önce ~73x), ViT-Tiny ~2.5–3.7x (`benchmarks/BENCH_VS_PYTORCH.md`).
- **PyTorch Paritesi:** 4 katmanlı `NewtonVisionCNN` forward çıktısında PyTorch 2.6.0 ile `8e-8` maksimum fark (`apps/pytorch_int8_benchmark.py`). Katman düzeyindeki 85 noktalı matriste en kötü FP32 farkı `2.7e-5` (hepsi tolerans içinde, `benchmarks/VERIFICATION_REPORT.md`).

### 3. Uygulama Demoları (`apps/`)
- `apps/malware_classifier.py`: İkili dosyaların ilk 1024 byte'ını 32x32 tensöre çevirip sınıflandıran CNN demosu. Model **sentetik** veriyle eğitilir (zararlı = NOP sled + rastgele byte, temiz = PE başlığı kalıbı); bu sentetik veri üzerindeki %100 doğruluk, gerçek zararlı yazılım tespit başarısını göstermez.
- `apps/image_predictor.py`: 323x323 JPEG dosyasını (`number.jpg`) MNIST formatına ön işleyip LeNet ile tahmin eden görsel boru hattı (tek görselde '5', %94.43 güven, ~2.3 ms).
- `apps/pytorch_int8_benchmark.py`: PyTorch vs newton-vision doğrudan başa baş doğruluk ve bellek kıyaslaması.

### 4. Framework Ergonomisi & Usability
- `Tensor.item()`, `Tensor.numpy()`, `len(Tensor)` standart metotları eklendi.
- `CrossEntropyLoss`, `MSELoss`, `SGD`, `Adam` sınıfları doğrudan root namespace'e (`nv.*`) aktarıldı.
- `LazyLinear` dinamik boyut eşleme ve katı boyut doğrulama muhafızları (shape guards) eklendi.
- `python -m unittest discover tests`: **94/94 test geçiyor** (MinGW build'inde 5 CUDA testi SKIP; MSVC + CUDA build'inde hepsi PASS) (TestCase dosyaları + `load_tests` ile toplanan fonksiyon testleri + C++ derleme testleri + eğitim bileşenleri + ONNX).

### 4.1 Eğitim Yeteneği (2026-10-23 doğrulaması)
- **Gerçek MNIST, LeNet:** 60 bin görsel, 3 epoch, Adam lr=1e-3 → 10 bin test görselinde **%98.43** (~80 sn/epoch).
- **ResNet-18 eğitim adımı:** Aynı ağırlık ve batch ile loss PyTorch'la birebir aynı (2.225944), BN running istatistikleri 6e-8 farkla eşleşiyor. Gradyan farkları PyTorch'un kendi float32 hatasıyla aynı büyüklükte (float64 referansıyla ölçüldü). Katman düzeyinde Conv+BN backward hatası ~1e-6.
- **ViT-Tiny / MobileNetV2:** Küçük batch'i ezberliyorlar (loss → ~1e-4); tüm parametreler güncelleniyor.
- **Eğitim adımı süresi (batch 2, 224², bu makine):** ResNet-18 7.36 sn → **1.16 sn**, MobileNetV2 5.26 sn → **1.03 sn** (2026-10-23 C++ conv backward optimizasyonu).

### 5. Açık Sorunlar (2026-10-23 doğrulaması)
- Python ve C++ `.nv.bin` anahtar adları ve Linear düzeni farklı (Python: `fc1.w` `(in, out)`, C++: `fc1.weight` `(out, in)`). Python'dan kaydedilen `.nv.bin` C++ tarafında doğrudan yüklenemez. Ters yön (C++ → Python) Python yükleyicisinin `weight → w` eşlemesiyle büyük olasılıkla çalışır, ancak test edilmedi.
- **Eğitim hızı:** PyTorch CPU'dan hâlâ birkaç kat yavaş (bkz. `benchmarks/BENCH_VS_PYTORCH.md`). Profil: süre C++ evrişim çekirdeklerinde; Python autograd payı ihmal edilebilir.
- **GPU ile eğitim yok.** CUDA build'i artık çalışıyor (`scripts/build_msvc_cuda.bat`: MSVC 14.44 + CUDA 12.4, GTX 1650'de doğrulandı). Ancak yalnızca 3 forward çekirdeği var (conv2d, relu, maxpool2d) ve her çağrıda host↔device kopyası yapılıyor; backward CPU'da. Gerçek GPU eğitimi için cihazda kalan tensörler ve tüm backward çekirdekleri gerekir; bu ayrı bir mimari projedir.
- **C++ başlık API'sinin CUDA dalları ölü kod:** `include/newton/layers.hpp`'deki `HAS_CUDA` dalları var olmayan `nv::cuda::*` fonksiyonlarını çağırıyor. Bu yüzden bağımsız C++ API'si yalnızca CPU olarak derleniyor.
- **Hâlâ olmayanlar:** alt-kelime (BPE) tokenizer, causal attention maskesi, ONNX'te dinamik batch boyutu ve kuantize katman dışa aktarımı.

### 6. Düzeltilen Sorunlar (2026-10-23)
- **Eğitim tutorial'ları grafı koparıyordu:** `tutorials/train_and_eval_lenet.py`, `tutorials/mnist_training.py`, `tutorials/01_train_mnist.py`, `examples/01_image_classification_lenet.py` ve `apps/image_predictor.py` (--train) düzleştirmeyi `nv.Tensor(x.data.reshape(...))` ile yapıyordu. Bu yüzden **konvolüsyon katmanları hiç eğitilmiyordu** (tam MNIST'te ağırlık değişimi 0.000000, doğruluk %96.2 yerine %98.4 olmalıydı). `Tensor.reshape` / `nv.Flatten` ile düzeltildi.
  - `tutorials/01_train_mnist.py` rastgele veri yerine gerçek MNIST kullanıyor ve INT8 deneyi için `nv.quantize_model`'i çağırıyor. Eski kod Linear'ı yanlış eksende kuantize ediyordu.
  - `train_and_eval_lenet.py` artık `lenet_mnist_v1.nv`'nin (image_predictor'ın modeli) üzerine yazmıyor, `lenet_mnist_trained.nv`'ye kaydediyor. lr 0.008 → 0.001: konvolüsyonlar da eğitilince 5. epoch'ta loss yükseliyordu.
- **Backward `RecursionError`:** Topolojik sıralama iteratif yapıldı (20 bin işlemlik zincir test edildi).
- **`Tensor.to()` gradyanı koparıyordu:** Eski düğümün closure'ını kopyalıyordu. Artık türevlenebilir bir kimlik işlemi.
- **Eklenenler** (hepsi PyTorch float64 referansıyla forward + backward doğrulandı):
  - Katmanlar: `Flatten`, `Conv1d`, `BatchNorm1d`, `Embedding`, `Upsample(nearest)`.
  - Kayıplar: `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`.
  - Eğitim yardımcıları: `clip_grad_norm_`, `StepLR`, `CosineAnnealingLR`, SGD/Adam `state_dict`/`load_state_dict`.
  - Veri artırma: `nv.transforms` (`Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`) ve `TensorDataset(..., transform=)`.
- **Gerçek ONNX dışa aktarımı** (`nv.export_onnx`, opset 17): İzleme tabanlı çalışıyor. ResNet-18, MobileNetV2 ve ViT onnxruntime ile doğrulandı. Grafı kopuk model ve desteklenmeyen katman açık hatayla reddediliyor. `onnx` opsiyonel bağımlılık: `pip install newton-vision[onnx]`.
- **`AdamW`** (torch.optim.AdamW ile aynı), **`Upsample` linear/bilinear** (`align_corners` her iki değer), **`nv.text.Tokenizer`** (kelime/karakter).
- **C++ evrişim hızlandırması:**
  - Backward döngüleri yeniden sıralandı ve paralelleştirildi (OpenMP + `omp simd`); `col2im` kanal bazında paralel.
  - Gruplu/depthwise evrişimde paralellik gruplar üzerine taşındı; iç içe paralel bölgeler `if(!omp_in_parallel())` ile engellendi.
  - Sonuç: eğitim adımı ResNet-18'de 6.3x, MobileNetV2'de 5.1x hızlandı. Parite 85/85 korundu.
- **MSVC + CUDA build'i** (`scripts/build_msvc_cuda.bat`, ayrı `build/msvc-cuda` klasörü):
  - **CUDA çekirdekleri:** Tüm CUDA çağrılarına hata kontrolü eklendi (hata olursa `RuntimeError`; önceden kontrol yoktu). Grid 1 boyutlu yapıldı: eski `grid.z = N*C_out` 65.535'i aşınca çekirdek sessizce hiç çalışmıyordu. `conv2d`'ye dilation/groups desteği eklendi; önceden yok sayılıyordu, gruplu evrişim CUDA'da yanlış sonuç verirdi.
  - **Taşınabilirlik:** `cpp/layers.h` eksik `#include <string>` hatası düzeltildi. OpenMP kullanımı 2.0 uyumlu hale getirildi (collapse → düz döngü, `omp simd` → makro).
  - **OpenMP seçimi:** MSVC'de `/openmp:llvm`, PyTorch'un Intel OpenMP'siyle aynı süreçte çakışıp süreci sonlandırdığı için klasik `/openmp` kullanılıyor.
  - **CUDA kapsamı:** CUDA yalnızca `_core` modülüne tanımlanıyor.
  - **Paket:** `newton_vision/__init__.py`, `CUDA_PATH`'ten `cudart` DLL dizinini ekliyor.
  - **Yan etki:** Düz döngüler MinGW build'inde de iş dağılımını iyileştirdi; eğitim adımı ResNet-18'de 1.16 → 0.88 sn, MobileNetV2'de 1.03 → 0.68 sn.
- **Git:** Proje klasöründe `main` dallı ayrı bir depo başlatıldı. `.gitignore`, uygulamaların ihtiyaç duyduğu modelleri ve demo görsellerini dahil edecek şekilde düzeltildi.
- **`AvgPool2d.forward` `None` döndürüyordu:** `return out` eklendi. ResNet-18 forward/backward çalışıyor, AvgPool gradyanı gradcheck ile doğrulandı.
- **`load_state_dict` yeniden yazıldı (tek implementasyon; `Module.load_state_dict` buna yönlendirir):**
  - Eski yükleyici yalnızca `weight/w/bias/running_*` adlarını tanıyordu. Bu yüzden `nv.save_model` → `nv.load_model` round-trip'inde ViT'in 200 anahtarından 100'ü, ResNet'in 102 anahtarından 40'ı (`gamma`, `beta`, `cls_token`, `pos_embed`…) **sessizce yüklenmiyordu**. Ayrıca kare Linear ağırlıkları yanlışlıkla transpoze ediliyordu.
  - Yeni yükleyici şekil uyuşmazlığında `ValueError` verir, `strict` parametresini uygular, eksik/beklenmeyen anahtarları `LoadResult` olarak döndürür ve uyarır. `load_model` / `load_model_bin` `strict=True` kullanır.
  - Doğrulama: ViT, ResNet-18, MobileNetV2 ve Linear için `.nv` ve `.nv.bin` round-trip'leri birebir.
- **`.nv.bin` round-trip Linear katmanını bozuyordu:** 2D ağırlıklar `(a, b, 1, 1)`, 3D tensörler `(1, N, 1, 1)` olarak geri geliyordu ve Linear forward çöküyordu. Yükleyici bu biçimleri artık hedef şekle geri açıyor.
- **torchvision ResNet-18 aktarımı:** `downsample.0/1` → `downsample.conv/bn` eşlemesi eklendi. `MaxPool2d`'ye `padding` parametresi eklendi ve ResNet stem'i torchvision ile aynı yapıldı (`padding=1`). Sonuç: max fark 6e-6, kosinüs 1.0 (önce 0.76).
- **Test toplama:** Fonksiyon tabanlı 5 test dosyası `load_tests` ile `unittest discover`'a bağlandı. `verify_real_training.py` dosyası `test_real_training.py` olarak yeniden adlandırıldı.
- **C++:** `tensor.hpp`'ye `<cstdint>` eklendi. `test_cpp_core.cpp` güncel `conv2d` imzasına uyarlandı. `tests/test_cpp_build.py` C++ self-test'i ve demoyu derleyip çalıştırıyor.
- **`bench_vs_pytorch.py`:** ViT-B/16 yerine aynı boyutlu torchvision ViT-Tiny kullanılıyor. newton-vision tarafı `eval + no_grad` ile ölçülüyor. Parametre sayıları eşleşmezse script duruyor. Warmup/Runs etiketleri koddan geliyor. Rapor elle yazılmış `BENCHMARK_RESULTS.md` yerine `BENCH_VS_PYTORCH.md`'ye yazılıyor.
- **Dokümantasyon:** `quantize_model()` yalnızca istatistik sözlüğü döndürür (katmanları yerinde değiştirir); dokümanlar buna göre düzeltildi.

---

## 🔒 Sürüm Yol Haritası (Version Roadmap)

- [x] **v1.0 (Tamamlandı):** PyPI yayını, Autograd motoru, Conv2d, OpenMP, MNIST eğitimi.
- [x] **v1.1 (Tamamlandı):** Maintainer kalitesi, `research/` klasörü, 5 Temel Soru README, `examples/` & `tutorials/` ayrımı.
- [x] **v1.1.1 - v1.1.6 (PyPI'da yayınlandı):** ViT-Tiny, MHSA, GELU, LayerNorm, ResNet-18, MobileNetV2, INT8 Kuantizasyon, PyTorch parite matrisi, `apps/` paketi. (Önceki sürümde bu satır yanlışlıkla "v1.2 - v1.5" olarak etiketlenmişti; PyPI'da 1.1.6'dan yüksek sürüm yoktur.)
- [ ] **v1.6:** Karma Hassasiyetli Eğitim (Mixed Precision AMP FP16).
- [ ] **v1.7:** JIT Graf Derleyicisi & Çekirdek Birleştirme (Kernel Fusion).
- [ ] **v2.0:** Dağıtık Eğitim & Multi-GPU.

---

*Son Güncelleme: 23 Ekim 2026 (dokümanlar kodla karşılaştırılarak düzeltildi)*
