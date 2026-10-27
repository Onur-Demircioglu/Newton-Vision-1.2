# newton-vision — Durum ve Yol Haritası

> Projenin güncel durumu için tek kaynak. Ne değiştiği: [CHANGELOG.md](CHANGELOG.md). Kullanım: [docs/guide.md](docs/guide.md).
> AI ajanları için mimari notlar ve tuzaklar: [AI_MEMORY.md](AI_MEMORY.md).

*Son güncelleme: 25 Ekim 2026*

---

## 📌 Özet

- **Sürüm:** `1.2.0` (bu dalda; PyPI'daki son sürüm 1.1.6). Değişiklikler: CHANGELOG → 1.2.0.
- **Ne:** C++17 çekirdekli (AVX2 + OpenMP), NumPy tabanlı autograd'a sahip, PyTorch benzeri API sunan eğitim/araştırma framework'ü.
- **Testler:** `python -m unittest discover tests` → 94 test. MinGW build'inde 5 CUDA testi atlanır; MSVC + CUDA build'inde hepsi geçer. PyTorch parite matrisi 85/85.
- **Build'ler:**
  - MinGW (varsayılan, kurulu olan; CPU'da en hızlı).
  - MSVC + CUDA (`scripts/build_msvc_cuda.bat`, [docs/cuda.md](docs/cuda.md)).

## ✅ Doğrulanmış Durum (2026-10-23 / 25)

| Alan | Durum |
|---|---|
| Eğitim | CIFAR-10 CNN test %86.44 (aynı koşullarda PyTorch %86.25). MNIST CNN %99.42 (PyTorch %99.47). ResNet-18 eğitim adımında loss PyTorch ile aynı. |
| Hız (CPU, batch 2, 224²) | Eğitim adımı ResNet-18 0.88 sn, MobileNetV2 0.68 sn. Çıkarımda PyTorch CPU'dan 2.5–4.3x yavaş ([docs/performance.md](docs/performance.md)). |
| PyTorch uyumu | Katman paritesi 85/85. torchvision ResNet-18 ağırlıklarıyla çıktı farkı 6e-6. |
| Dışa aktarım | `.nv`, `.nv.bin` ve ONNX (opset 17). ONNX çıktısı onnxruntime ile doğrulanıyor. |
| CUDA | `conv2d`, `relu`, `maxpool2d` forward GTX 1650'de CPU ile aynı sonucu veriyor. |
| Demolar | `apps/image_predictor.py` (`number.jpg` → '5', %94.43). `apps/malware_classifier.py` yalnızca **sentetik** veriyle eğitilmiş bir demo. `apps/pytorch_int8_benchmark.py`. |

## ⚠️ Açık Sorunlar ve Sınırlar

- **GPU ile eğitim yok.** CUDA'da yalnızca 3 forward çekirdeği var, her çağrıda host↔device kopyası yapılıyor ve backward CPU'da. Gerçek GPU eğitimi için cihazda kalan tensörler ve tüm backward çekirdekleri gerekir; bu ayrı bir mimari proje.
- **C++ başlık API'sinin `HAS_CUDA` dalları ölü kod:** Var olmayan `nv::cuda::*` fonksiyonlarını çağırıyor. Bağımsız C++ API'si bu yüzden yalnızca CPU olarak derleniyor.
- **`.nv.bin` uyumsuzluğu:** Python ve C++ `.nv.bin` anahtar adları ve Linear ağırlık düzeni farklı (`fc1.w` `(in, out)` ile `fc1.weight` `(out, in)`). Python'dan kaydedilen dosya C++'ta doğrudan yüklenemiyor.
- **MSVC build'inin CPU yolu** MinGW'den yavaş, çünkü MSVC'de `omp simd` yok.
- **Eksik özellikler:** BPE tokenizer, causal attention maskesi, ONNX'te dinamik batch ve kuantize katman dışa aktarımı, metrik paketi (Accuracy/F1/ConfusionMatrix yardımcıları).
- **Git geçmişi:** İlk commit (`99dc0f8`) kişisel not dosyası `ileriye_dönük.txt`'i içeriyor. Dosya artık takip edilmiyor ama geçmişte duruyor; GitHub'a push etmeden önce geçmişten temizlenmesi düşünülmeli.

## 🗺️ Yol Haritası

**Hedef nişler** (vizyon, henüz gerçekleşmedi):
1. Ağır Python bağımlılıkları olmadan gömülü sistemlerde (Raspberry Pi, akıllı kamera) çalışan hafif bir C++ çıkarım motoru.
2. EDR ajanlarına gömülebilen, ikili dosyaları tensöre çevirip yerel olarak sınıflandıran bir tehdit analiz çekirdeği. Mevcut malware demosu bunun sadece sentetik bir prototipi.

| Faz | İçerik | Durum |
|---|---|---|
| **v1.2 Sertleştirme** | PyTorch parite matrisi, gradcheck, `clip_grad_norm_`, LR scheduler, veri artırma | ✅ Tamamlandı (yayınlanmadı) |
| | INT8 doğruluk kaybı ölçümü | 🟡 Kısmen (`tutorials/01_train_mnist.py`: FP32 %89.70, INT8 %89.80) |
| | Metrik paketi | ⬜ Yok |
| **v1.3 Donanım** | ARM NEON SIMD, WebAssembly, FP16 | ⬜ |
| | GPU eğitimi: cihazda kalıcı tensörler + backward çekirdekleri | ⬜ (CUDA build'i ve 3 forward çekirdeği hazır) |
| **v1.4 Nesne tespiti** | Tiny-YOLO başlığı, C++ NMS, video akışı | ⬜ |
| **v2.0 C++ SDK** | Bağımsız C++ SDK (CUDA dahil), PE/ELF ayrıştırıcı, hafif mikroservis | ⬜ |

## 📏 Mühendislik İlkeleri

1. Ölçülmemiş çarpan ("4x", "10x") gerçek gibi yazılmaz.
2. Her hız verisi donanım, model, çözünürlük, batch ve kapsamla birlikte verilir.
3. Her değişiklikten sonra `python -m unittest discover tests` ve `python benchmarks/run_all.py` geçmelidir.
4. Kural ve ilkeler: [CLAUDE.md](CLAUDE.md), [PROJECT_RULES.md](PROJECT_RULES.md).
