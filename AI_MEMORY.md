# newton-vision — AI Mimari Hafızası

> Projeyi devralan AI ajanları ve geliştiriciler için: mimari, kurallar ve yaşanmış tuzaklar.
> Güncel durum: [PROGRESS.md](PROGRESS.md) · Değişiklikler: [CHANGELOG.md](CHANGELOG.md) · Kurallar: [PROJECT_RULES.md](PROJECT_RULES.md), [CLAUDE.md](CLAUDE.md)
> Son güncelleme: 2026-10-25

---

## 1. Mimari Özeti

- **Python (`src/newton_vision/`):** `Tensor` ve autograd grafı, modüller, Linear/MHSA matris çarpımları (NumPy `@`), modeller, veri, API.
- **C++ (`cpp/`, pybind11 → `newton_vision._core`):** conv2d (im2col + AVX2 GEMM) forward/backward, pooling, aktivasyonlar, BN/LN, OpenMP.
- **CUDA (`cpp/cuda_layers.cu`):** Yalnızca MSVC + CUDA build'inde; conv2d/relu/maxpool2d forward. Her çağrıda host↔device kopyası yapılır.
- **Autograd:** Reverse-mode, iteratif topolojik sıralama. Tensör verisi her zaman NumPy `float32` (host belleği); `device` yalnızca hangi çekirdeğin çağrılacağını seçer.
- **Profil gerçeği:** Eğitim süresinin neredeyse tamamı C++ `conv2d`/`conv2d_backward` içinde geçer; Python autograd payı ihmal edilebilir.

| Modül | Dosya | Sorumluluk |
|---|---|---|
| `tensor` | `tensor.py` | `Tensor`, autograd, `no_grad`, `_add_grad` (unbroadcast) |
| `module` | `module.py` | `Module`, `Parameter`, `_parameters/_modules/_buffers`, `state_dict` |
| `containers` | `containers.py` | `Sequential`, `ModuleList` |
| `layers` | `layers.py` | Tüm katmanlar (Conv1d/2d, Linear, BN1d/2d, LN, MHSA, pool, Upsample, Embedding, Flatten, Dropout, aktivasyonlar) |
| `gravity` | `gravity.py` | Kayıplar, SGD/Adam/AdamW (`state_dict` destekli), StepLR/CosineAnnealingLR, `clip_grad_norm_` |
| `data`, `transforms`, `text` | | Dataset/DataLoader/ImageFolder/CSVDataset, veri artırma, Tokenizer |
| `weights` | `weights.py` | `load_state_dict`: tek implementasyon (`Module.load_state_dict` buna yönlendirir) |
| `jit`, `onnx_export` | | `.nv` / `.nv.bin`, `InferenceEngine`, izleme tabanlı ONNX (opset 17) |
| `quantization` | `quantization.py` | Per-channel INT8 ağırlık kuantizasyonu (hız kazancı yok) |
| `models/` | `resnet.py`, `mobilenet.py`, `vit.py` | Model zoo (LeNet yalnızca `tutorials/` ve `apps/` içinde) |

---

## 2. Kalıcı Kurallar (yaşanmış hatalardan)

1. **Grafı koparma:** Ara tensörü asla `.data` üzerinden yeni bir `nv.Tensor(...)`'a sarma; `nv.Flatten()` / `Tensor.reshape` kullan. Bu hata yüzünden tutorial'larda konvolüsyonlar hiç eğitilmiyordu. Eğitim örneklerinde ilk katmanın ağırlığının değiştiğini kontrol et.
2. **`forward()` her zaman çıktıyı `return` etmeli:** `AvgPool2d` `None` döndürdüğü için ResNet-18 çöküyordu.
3. **Testler toplanmalı:** Modül düzeyinde `test_*` fonksiyonu kullanan dosyaya `load_tests` bloğunu ekle; aksi halde `unittest discover` dosyayı çalıştırmaz. AvgPool hatası bu yüzden gözden kaçmıştı.
4. **Ağırlık yükleme:**
   - Yeni bir parametre adı eklerken `nv.save_model → nv.load_model` round-trip'inin birebir olduğunu test et. Eski yükleyici `gamma`, `beta`, `cls_token` gibi adları sessizce atlıyordu.
   - Kare Linear ağırlıklarında yönü anahtar adı belirler: `weight` = PyTorch `(out, in)`, `w` = newton-vision `(in, out)`.
   - `Embedding` ağırlığının adı bilerek `weight`; `w` adı Linear'a özgü transpoz kuralını tetiklerdi.
5. **Hata yutma yok:** Şekil uyuşmazlığı `ValueError`, CUDA hataları `RuntimeError`, ONNX'te desteklenmeyen katman `NotImplementedError` vermeli.
6. **Ölçmeden yazma:** Benchmark'ta iki taraf aynı boyutta model ve aynı modla (`eval` + `no_grad`) ölçülmeli. Eski "ViT 2.28x hızlı" sonucu ViT-B/16 ile karşılaştırmaktan geliyordu.
7. **OpenMP 2.0 uyumu (MSVC):** `collapse` kullanma, düz döngü yaz; `omp simd` yerine `NV_OMP_SIMD` / `NV_OMP_SIMD_SUM` makrolarını kullan. İç içe paralel bölgeleri `if(!omp_in_parallel())` ile engelle.
8. **Güvenli deserialization:** `np.load(..., allow_pickle=False)`.
9. **Windows konsolu:** CLI çıktıları cp1254'te çökmemeli; testlerde `PYTHONIOENCODING=utf-8` kullan.

---

## 3. Derleme Notları

- **MinGW (varsayılan):**
  - `build/cp313-cp313-win_amd64` içinde `cmake --build . --target _core` çalıştır; PATH'te `/c/msys64/mingw64/bin` (GCC 15) olmalı, PATH'teki eski gcc 6.3'ü kullanma.
  - Editable kurulum `.pyd`'yi site-packages'taki `newton_vision/` klasöründen yükler; derlenen `_core*.pyd`'yi oraya kopyala.
- **MSVC + CUDA:** `scripts/build_msvc_cuda.bat` (MSVC 14.44 + CUDA 12.4 + SDK 10.0.22621.0, sm_75), ayrıntılar [docs/cuda.md](docs/cuda.md).
  - MSVC 14.51 ile CUDA 12.4'ün `cudafe++` aracı çöküyor.
  - Bu makinede vcvars SDK `um`/`shared` yollarını eklemiyor; script bunları elle ekliyor.
  - `/openmp:llvm` kullanma: PyTorch'un `libiomp5md`'siyle çakışıp süreci sonlandırıyor (OMP Error #15).
- **Kurulu paketi değiştirmeden başka bir `.pyd` test etmek:** Editable kurulumun `ScikitBuildRedirectingFinder`'ını `sys.meta_path`'ten çıkar; aksi halde kurulu `.pyd` yüklenir.

---

## 4. Yeni Özellik Eklerken

1. **Katman:** `Module`'den türet, eğitilebilir tensörleri `Parameter` yap, `self.training`'i doğru kullan, çıktıyı `return` et.
2. **Tensör işlemi:** `tensor.py`'de `_backward` closure'ı tanımla; gradyanı `_add_grad` ile biriktir.
3. **ONNX:** Yeni katmana `onnx_export._CONVERTERS` içinde bir dönüştürücü ekle; yoksa dışa aktarım `NotImplementedError` verir.
4. **Doğrula:**
   - PyTorch (float64) referansına karşı forward + backward testi yaz (`tests/test_training_components.py` örnek).
   - `python -m unittest discover tests` (94 test) ve `python benchmarks/run_all.py` geçmeli.

---

## 5. Mimari Sınırlar

- **GPU:** GPU ile eğitim yok; cihazda kalıcı tensör, caching allocator ve backward çekirdekleri gerekir.
- **Ölçek:** LLM ölçeği yok (causal mask, BPE, FlashAttention, mixed precision, dağıtık eğitim eksik).
- **Dizi modelleri:** RNN/LSTM katmanı yok.
- **C++ başlık API'si:** Yalnızca CPU; `HAS_CUDA` dalları var olmayan fonksiyonlara başvuruyor.
