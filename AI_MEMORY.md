# NEWTON-VISION — AI KOD VE MİMARİ HAFIZA DÖKÜMANI (AI MEMORY)

> **Belge Tipi:** Canlı Sistem & Mühendislik Hafızası (AI Memory Core)  
> **Framework:** `newton-vision` (v1.1.6)  
> **Konum:** `c:\Users\onur_\Desktop\newton-vision`  
> **İlişkili Belgeler:** [`PROJECT_RULES.md`](file:///c:/Users/onur_/Desktop/newton-vision/PROJECT_RULES.md) (Anayasa & Sıfır Hata Protokolü) | [`ROADMAP.md`](file:///c:/Users/onur_/Desktop/newton-vision/ROADMAP.md) (Gelecek Yol Haritası) | [`MATHEMATICS.md`](file:///c:/Users/onur_/Desktop/newton-vision/MATHEMATICS.md) (Matematiksel Şartname)  
> **Son Güncelleme:** 2026-10-23 (iddialar kodla karşılaştırılıp düzeltildi; bkz. Bölüm 8)  
> **Rol / Bakış Açısı:** Senior Software Developer & Senior Cyber Security Specialist (15+ Yıl Deneyim)  
> **Hedef:** Bu döküman, projeyi devralacak herhangi bir AI ajanının veya kıdemli yazılım mühendisinin projenin tüm tarihçesini, mimari kararlarını, çözülmüş gizli hatalarını (gotchas), sınırlarını ve çalışma prensiplerini sıfır bilgi kaybıyla anlaması için tasarlanmıştır.

---

## 1. PROJE ÖZETİ VE ÇEKİRDEK KONSEPT

`newton-vision`, harici ağır bağımlılıklara (PyTorch runtime vb.) ihtiyaç duymadan derin öğrenme modellerini eğitmek, kuantize etmek ve düşük gecikmeyle (low-latency) çalıştırmak amacıyla sıfırdan C++17 ve Python C-API (PyBind11 / ctypes) ile inşa edilmiş bir **Derin Öğrenme ve Bilgisayarlı Görü Motorudur**.

* **Dil Mimarisi:** Ağır katman çekirdekleri (conv2d im2col+GEMM, pooling, aktivasyonlar, BN/LN forward ve çoğu backward), AVX2 GEMM ve CUDA rutinleri C++17 (`cpp/`, pybind11 ile `newton_vision._core`). Tensor sınıfı, autograd grafı, Linear/MHSA matris çarpımları (NumPy `@`), modeller, veri yükleyiciler ve API katmanı Python (`src/newton_vision/`).
* **Autograd:** Reverse-mode otomatik türev (DAG tabanlı dinamik hesaplama grafı, Python/NumPy üzerinde; topolojik sıralama özyinelemeli).
* **Cihaz:** Tensör verisi her zaman NumPy (host belleği). `device="cuda"` yalnızca CUDA build'inde conv2d/relu/maxpool forward'u GPU'ya yönlendirir; mevcut yerel build ve CI wheel'leri CUDA'sız.
* **Parite:** Katman düzeyinde PyTorch 2.6.0 ile 85/85 ölçüm noktası tolerans içinde (en kötü FP32 farkı ~2.7e-5); 4 katmanlı `NewtonVisionCNN` forward'unda 8e-8, torchvision ResNet-18 aktarımında 6e-6. Ağırlık aktarımı isim eşlemesine dayanır; eşleşmeyen anahtarlar uyarı verir (`strict=True` ile hata), şekil uyuşmazlığı her zaman hata verir.
* **Bellek & Dağıtım:** FP32 -> INT8 simetrik per-channel ağırlık kuantizasyonu (~%75 ağırlık belleği tasarrufu, hız kazancı yok), serileştirme (`.nv` = sıkıştırılmış NumPy, `.nv.bin` = C++ ikili).

---

## 2. KRONOLOJİK İLERLEME VE GELİŞİM ZAMAN ÇİZELGESİ

```
[FAZ 1: Çekirdek Motor] 
   └── C++17 Tensor motoru, GEMM, im2col konvolüsyon, temel ters türev (backward).
[FAZ 2: Model & Katman Genişlemesi]
   └── Conv2d, Linear, BatchNorm2d, LayerNorm, GELU, MaxPool2d, AvgPool2d.
   └── LeNet-5, ResNet-18, MobileNetV2, ViT-Tiny (Vision Transformer).
[FAZ 3: Kuantizasyon & SIMD Hızlandırma]
   └── INT8 per-channel kuantizasyon motoru, QuantizedLinear, QuantizedConv2d.
   └── AVX2 SIMD vektörizasyonu ile CPU çıkarımında 500-800 FPS hız.
[FAZ 4: PyTorch Paritesi & Serialization]
   └── NumPy'a çevrilmiş PyTorch state_dict'ini yükleyen isim eşlemeli yükleyici (.pt/.pth dosyasını doğrudan okumaz).
   └── Standalone C++ başlıkları (include/newton/), Python InferenceEngine, izleme tabanlı gerçek ONNX dışa aktarımı (`onnx_export.py`, onnxruntime ile doğrulanır).
[FAZ 5: Kritik Hata Ayıklama & Mimari Sertleştirme]
   └── Sequential katmanının Module olmayan aktivasyonları atlaması çözüldü.
   └── MobileNetV2 parametresiz kalma sorunu çözüldü (ModuleList entegrasyonu).
   └── LazyLinear ve dinamik tensör boyut doğrulayıcıları eklendi.
[FAZ 6: Somut Uygulama Paketi (apps/)]
   └── apps/malware_classifier.py: İkili dosyaları 2D tensöre çevirip sınıflandıran CNN demosu (sentetik veriyle eğitilir).
   └── apps/image_predictor.py: 323x323 JPEG çizimi işleyip doğru tahmin eden görüntü motoru.
   └── apps/pytorch_int8_benchmark.py: PyTorch 2.6 ile birebir sayısal parite testi.
[FAZ 7: Ergonomi & Usability Sertleştirmesi]
   └── Tensor.item(), Tensor.numpy(), len(Tensor) eklendi.
   └── Root namespace export'ları (CrossEntropyLoss, Adam, SGD) tamamlandı.
   └── unittest discover: 89/89 (2026-10-23; bkz. Bölüm 8).
```

---

## 3. MODÜL VE DİZİN HARİTASI (ARCHITECTURAL TAXONOMY)

| Modül Adı | Dosya Yolu | Sorumluluk & Kritik Bileşenler |
|---|---|---|
| **`tensor`** | `src/newton_vision/tensor.py` | `Tensor` sınıfı, reverse-mode Autograd, gradyan biriktirme (`_add_grad`), dilimleme (`__getitem__` with `np.add.at`), `.item()`, `.numpy()`, `no_grad()`, `enable_grad()`. |
| **`module`** | `src/newton_vision/module.py` | Temel `Module`, `Parameter`, `_parameters`, `_modules`, `_buffers` takibi, `__getattr__` dinamik çözümleme, `state_dict()` ve `load_state_dict()`. |
| **`containers`** | `src/newton_vision/containers.py` | `Sequential` (sıralı katman zinciri), `ModuleList` (indekslenebilir ve parametre toplayan dinamik modül listesi). |
| **`layers`** | `src/newton_vision/layers.py` | `Conv2d`, `Linear`, `LazyLinear`, `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, `GELU`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d`, `LayerNorm`, `MultiHeadAttention`, `Dropout`, `Flatten`, `Conv1d`, `BatchNorm1d`, `Embedding`, `Upsample`. |
| **`gravity`** | `src/newton_vision/gravity.py` | Kayıplar (`MSELoss`, `CrossEntropyLoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`), optimizatörler (`SGD`, `Adam`, `state_dict` destekli), `StepLR`, `CosineAnnealingLR`, `clip_grad_norm_`. |
| **`data`** | `src/newton_vision/data.py` | Veri hattı: `Dataset`, `TensorDataset`, `DataLoader` (batching + shuffle), `ImageFolder` (klasör tabanlı görsel okuma), `CSVDataset` (tabüler veri). |
| **`transforms`** | `src/newton_vision/transforms.py` | Örnek başına veri artırma: `Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`. |
| **`models`** | `src/newton_vision/models/` (`resnet.py`, `mobilenet.py`, `vit.py`) | `resnet18()`, `mobilenet_v2()`, `vit_tiny()`, `vit_base()`. (LeNet yalnızca `tutorials/` ve `apps/` içinde tanımlı.) |
| **`quantization`**| `src/newton_vision/quantization.py`| FP32 <-> INT8 simetrik ve per-channel kuantizasyon, `quantize_model()`, `QuantizedLinear`, `QuantizedConv2d`. |
| **`weights`** | `src/newton_vision/weights.py` | PyTorch ağırlık aktarıcısı; Linear katmanlarında `(out, in) -> (in, out)` transpozisyonunu otomatik çözen akıllı yükleyici. |
| **`jit`** | `src/newton_vision/jit.py` | `.nv` (`np.savez_compressed`) ve `.nv.bin` (özel ikili format) serileştirme, `InferenceEngine`, `export_onnx` (gerçek ONNX, `onnx_export.py`). |
| **`apps`** | `apps/` | Demo senaryoları (sentetik veriyle malware sınıflandırma, görsel tahmin, PyTorch INT8 kıyaslaması). |

---

## 4. ÇÖZÜLMÜŞ KRİTİK HATALAR VE "GOTCHAS" (AI DEBUGGING LOG)

Gelecekte kod yazarken kesinlikle tekrarlanmaması gereken, bizzat yaşanmış ve çözülmüş teknik tuzaklar:

### 1. `Sequential` Aktivasyon Kaybı Hatası
* **Kök Neden:** İlk sürümlerde `ReLU`, `Sigmoid`, `MaxPool2d` gibi sınıflar `Module` yerine düz Python callable fonksiyonuydu. `Sequential.__init__` ise yalnızca `isinstance(x, Module)` olanları `_modules` sözlüğüne kaydediyordu. Sonuç olarak katmanlar zincirden siliniyor veya ağırlıksız ara katmanlar yürütülürken hata veriyordu.
* **Çözüm:** Tüm aktivasyonlar ve havuzlama (pooling) sınıfları `Module` sınıfından türetildi ve standart `forward()` imzasına kavuşturuldu.

### 2. `MobileNetV2` Parametresiz Kalma (0 Parameters) Hatası
* **Kök Neden:** `InvertedResidual` blokları `self.conv = [Conv2d, BatchNorm, ReLU]` şeklinde standart Python `list` yapısında tutuluyordu. `Module.parameters()` özyineli araması Python listelerinin içine inmediği için `model.parameters()` çağrıldığında 0 parametre dönüyordu.
* **Çözüm:** `ModuleList` konteynırı geliştirildi; hem liste indekslemesini (`self[i]`) destekler hale getirildi hem de `_modules` üzerinden parametreleri kusursuz topladı.

### 3. PyTorch vs newton-vision Linear Matris Transpoz Farkı
* **Kök Neden:** PyTorch Linear katmanında ağırlık matrisini `(out_features, in_features)` boyutunda tutar ($y = xA^T + b$). `newton-vision` ise GEMM optimizasyonu için doğrudan `(in_features, out_features)` tutar ($y = xA + b$). PyTorch'tan ağırlık yüklerken matris transpoze edilmezse boyut uyuşmazlığı patlıyordu.
* **Çözüm:** `newton_vision.weights.load_state_dict` fonksiyonuna akıllı boyut kontrolü eklendi; eğer `w.shape == (out, in)` ise otomatik `.T` (transpoz) alarak hatayı tamamen şeffaf hale getirdi.
* **2026-10-23 yeniden yazım:** Eski yükleyici yalnızca `weight/w/bias/running_*` adlarını tanıyor, geri kalanını sessizce atlıyordu. Bu yüzden `save_model → load_model` round-trip'i ViT'te parametrelerin yarısını kaybediyordu. Ayrıca kare Linear ağırlıklarını yanlışlıkla transpoze ediyordu. Yeni kurallar:
  * Önce tüm anahtarlar doğrulanır, sonra yazılır.
  * Şekil uyuşmazlığında `ValueError`.
  * `strict` uygulanır; `LoadResult(missing_keys, unexpected_keys)` döner.
  * PyTorch adları eşlenir: `weight → w/gamma`, `bias → beta`, `blocks.0 → blocks_0`, `downsample.0 → 0. alt modül`.
  * Kare Linear'da yönü anahtar adı belirler: `weight` = `(out, in)`, `w` = `(in, out)`.
  * `.nv.bin`'in `(a, b, 1, 1)` ve `(1, N, 1, 1)` biçimleri hedef şekle geri açılır.
* **Kural:** Yeni bir parametre adı eklerken `nv.save_model → nv.load_model` round-trip'inin birebir olduğunu test edin.

### 4. Windows Konsol Çökmesi (`charmap` cp1254 Encoding)
* **Kök Neden:** `nv.help()` komutu Windows Türkçe konsolunda (cp1254 kod sayfası) çalışırken Unicode kutu çizim karakterleri (`─`, `│`, `┌`) nedeniyle `UnicodeEncodeError: 'charmap' codec can't encode character` fırlatıyordu.
* **Çözüm:** Tüm CLI ve terminal görsel çıktıları UTF-8 / ASCII-safe formatına (`+`, `-`, `|`, `*`) uyarlandı.

### 5. `Tensor.item()` ve Root Namespace İzolasyonu
* **Kök Neden:** Standart PyTorch geliştiricilerinin `loss.item()` refleksi `newton-vision` üzerinde `AttributeError` veriyordu. Benzer şekilde `nv.CrossEntropyLoss` yerine `nv.gravity.CrossEntropyLoss` yazılması zorunluluğu framework ergonomisini bozuyordu.
* **Çözüm:** `Tensor.item()`, `Tensor.numpy()`, `len(Tensor)` metotları eklendi; `CrossEntropyLoss`, `MSELoss`, `SGD`, `Adam` sınıfları doğrudan en üst namespace'e (`import newton_vision as nv -> nv.Adam`) taşındı.

### 6. Serileştirme Güvenliği & RCE Koruması (Safe Deserialization)
* **Kök Neden:** `jit.load_model` içinde `np.load` doğrudan çağrıldığında `pickle` mekanizması devreye girebilirdi. Kötü niyetli bir saldırgan özel olarak hazırlanmış bir `.nv` ağırlık dosyası ile Arbitrary Code Execution (RCE) çalıştırabilirdi.
* **Çözüm:** `np.load(filepath, allow_pickle=False)` parametresi zorunlu kılınarak Python bytecode/pickle çalıştırma vektörü tamamen kapatıldı.

### 7. Regularization ve Optimizer Eksiklikleri (Dropout & Weight Decay)
* **Kök Neden:** Aşırı öğrenmeyi (overfitting) engelleyen `Dropout` katmanı ve ağırlık patlamasını önleyen L2 cezası (`weight_decay`) optimizatörlerde yoktu.
* **Çözüm:** Inverted Dropout (`nv.Dropout(p=0.5)`) eklendi (`train` modunda rastgele sıfırlama ve ölçekleme, `eval` modunda identity pass). `SGD` ve `Adam` optimizatörlerine PyTorch standardında `weight_decay` desteği getirildi.

---

## 5. DOĞRULANMIŞ PERFORMANS VE TEST VERİLERİ (BENCHMARKS)

| Metrik / Test | Ölçülen Değer | Durum |
|---|---|---|
| **Birim Testler (`unittest discover`)** | **89 / 89 Başarılı** (~20 sn): TestCase dosyaları, `load_tests` ile toplanan fonksiyon testleri, C++ derleme testleri, eğitim bileşenleri (PyTorch float64 ile) | Yeşil (2026-10-23) |
| **torchvision ResNet-18 ağırlık aktarımı** | Max fark `6e-6`, kosinüs `1.0` (rastgele ağırlıklar, BN istatistikleri rastgele) | Doğrulandı (2026-10-23) |
| **PyTorch 2.6.0 Parite — katman matrisi** | 85/85 nokta tolerans içinde, en kötü FP32 farkı `2.7e-5` | Doğrulandı (2026-10-23) |
| **PyTorch 2.6.0 Parite — 4 katmanlı CNN forward** | Maksimum fark `8e-8`, kosinüs `%100.000000` | Doğrulandı (2026-10-23) |
| **INT8 Kuantizasyon Tasarrufu** | ResNet-18: 43.64 MB ➔ 10.92 MB (**%74.96 ağırlık belleği**); hız kazancı yok | Doğrulandı |
| **CPU Çıkarım Hızı** | 4 katmanlı CNN `(1,3,32,32)`: ~540 – 780 FPS. 224² girdide PyTorch CPU'ya göre: ResNet-18 ~4.2x, MobileNetV2 ~4.3x (optimizasyondan önce ~73x), ViT-Tiny ~2.5–3.7x yavaş (`benchmarks/BENCH_VS_PYTORCH.md`; mutlak süreler koşudan koşuya değişken) | Ölçüldü |
| **Gerçek Görüntü Testi (`number.jpg`)**| Tek bir 323x323 JPEG, MNIST merkezleme ile **%94.43 güvenle '5'** | ~2.3 ms |
| **Malware Sınıflandırma (`apps/`)**| **Sentetik** veri üzerinde %100 doğruluk; gerçek zararlı yazılım veri setinde ölçülmedi | Demo |
| **Derin Model Geriye Türev (Backward)**| ResNet-18 (62/62 parametrede gradyan, `grad_norm ≈ 33.7`, seed=0, 10 sınıf, 2026-10-23), MobileNetV2 (`grad_norm: 83.5`), ViT-Tiny | Başarılı |

---

## 6. MİMARİ SERT SINIRLAR (ÇERÇEVE NEREDE ÇALIŞMAZ?)

Bu framework'ü kullanırken aşılmaması gereken fiziksel ve mimari limitler:

1. **Dev Dil Modelleri (LLMs - 1B+ Parametre):** Framework'te PyTorch'un C++ Caching VRAM Allocator'ı veya FlashAttention mekanizması bulunmamaktadır. Trilyonluk token eğitimleri için uygun değildir; VRAM fragmentasyonu veya RAM yetersizliği yaşanır.
2. **Dinamik Döngüsel Ağlar (RNN / LSTM):** Hesaplama grafı Directed Acyclic Graph (DAG) ve Residual/Skip bağlantılar için kurgulanmıştır; döngüsel (cyclical recurrent) BPTT (Backpropagation Through Time) döngüleri için optimize edilmiş dahili LSTM katmanı yoktur.
3. **Çoklu Makine Dağıtık Eğitim (Multi-Node DDP):** Tek makine çok çekirdek (OpenMP/SIMD) ve tek CUDA GPU hedefli çalışır; makineler arası NCCL soket senkronizasyonu içermez.

---

## 7. YENİ BİR Aİ İÇİN HIZLI BAŞLANGIÇ TALİMATI

Eğer projede yeni bir özellik geliştirecekseniz:
1. Katman ekleyecekseniz: Mutlaka `newton_vision.module.Module` miras alınmalı, parametreler `Parameter` olarak atanmalı ve `forward()` sonunda çıktı tensörü `return` edilmelidir.
2. Tensör operasyonu ekleyecekseniz: `src/newton_vision/tensor.py` içine `_backward` fonksiyonu tanımlanarak topological sort grafına eklenmelidir.
3. Testleri koşmak için: `python -m unittest discover tests` komutunu çalıştırın (89 test). Modül düzeyinde `test_*` fonksiyonu kullanan yeni bir test dosyası eklerseniz, dosyaya aynı `load_tests` bloğunu ekleyin. Aksi halde unittest o dosyayı **toplamaz**; AvgPool2d hatası tam olarak bu yüzden gözden kaçmıştı.

---

## 8. 2026-10-23 DOKÜMAN DENETİMİ — DÜZELTİLEN YANLIŞ İDDİALAR

Aşağıdaki iddialar kodla karşılaştırıldı ve yanlış bulundu; dokümanlarda tekrar kullanılmamalıdır:
* **"ViT-Tiny PyTorch'tan 2.28x hızlı":** `bench_vs_pytorch.py` PyTorch tarafında ViT-B/16 (86M) kullanıyordu. Aynı mimaride newton-vision ~3x yavaş.
* **"CUDA VRAM'e aktarır":** `.to("cuda")` yalnızca cihaz etiketini değiştirir; veri NumPy'da kalır.
* **"ONNX export":** O tarihte gerçek ONNX üretilmiyordu (aynı gün izleme tabanlı gerçek dışa aktarım eklendi).
* **"INT8 SIMD ile hızlı çıkarım":** INT8 yalnızca bellek optimizasyonu; C++ `gemm_int8_cpu` bağlı değil.
* **"Gerçek PE dosyalarında %100":** Eğitim/test verisi sentetik.
* **"Peer-reviewed":** Bağımsız bir hakem denetimi yapılmadı.
* **README Senaryo 1 "Beklenen Çıktı":** Sayılar uydurmaydı; gerçek çıktı ile değiştirildi.
* **"tests/test_cpp_core.cpp PASS":** Dosya güncel API ile derlenmiyordu (sonradan düzeltildi, artık `test_cpp_build.py` ile otomatik test ediliyor).

Aynı gün düzeltilen kod hataları (ayrıntılar `PROGRESS.md` → "Düzeltilen Sorunlar"):
* `AvgPool2d.forward` `return` eksikti.
* `load_state_dict` sessizce parametre kaybediyordu.
* `.nv.bin` round-trip Linear'ı bozuyordu.
* `MaxPool2d`'de padding yoktu, bu yüzden ResNet stem'i torchvision'dan sapıyordu.
* Fonksiyon testleri unittest'e bağlı değildi.
* `tensor.hpp`'de `<cstdint>` eksikti.
* Benchmark adil değildi: ViT-B/16 ile karşılaştırıyordu ve newton-vision tarafında `eval`/`no_grad` yoktu.
* **Eğitim tutorial'ları grafı koparıyordu:** `nv.Tensor(x.data.reshape(...))` ile düzleştirildiği için konvolüsyonlar hiç eğitilmiyordu.
* **Kalıcı kural:** Ara tensörü asla `.data` üzerinden yeni bir `Tensor`'a sarmayın, bu grafı koparır. `nv.Flatten()` / `Tensor.reshape` kullanın. Eğitim örneklerinde ilk katmanın ağırlığının değiştiğini kontrol edin (`tutorials/01_train_mnist.py` bunu yapar).
* Backward özyinelemeliydi (RecursionError). `Tensor.to()` gradyanı koparıyordu.
* Eklenenler: `Flatten`, `Conv1d`, `BatchNorm1d`, `Embedding`, `Upsample`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`, `clip_grad_norm_`, `StepLR`, `CosineAnnealingLR`, optimizer `state_dict`, `nv.transforms`. Testler: `tests/test_training_components.py`.
* Sonradan eklenenler: `AdamW`, `Upsample(linear/bilinear)`, `nv.text.Tokenizer`, gerçek ONNX (`onnx_export.py`).
* **Performans gerçeği:** Eğitimde süre neredeyse tamamen C++ `conv2d`/`conv2d_backward` içinde geçer; Python autograd payı ihmal edilebilir. Backward eskiden tamamen seri ve naifti.
* **C++ değişikliğinden sonra:** `build/cp313-cp313-win_amd64` içinde `cmake --build . --target _core` çalıştırın (PATH'te `/c/msys64/mingw64/bin` olmalı). Ardından `_core*.pyd` dosyasını site-packages'taki `newton_vision/` klasörüne kopyalayın; editable kurulum `.pyd`'yi oradan yüklüyor.
* **GPU:** CUDA 12.4 ile MSVC 14.51 uyumsuz (`cudafe++` çöküyor). CUDA 12.9+ veya VS 2022 Build Tools gerekir.
