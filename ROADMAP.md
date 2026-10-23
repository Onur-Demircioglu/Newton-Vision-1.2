# NEWTON-VISION — STRATEJİK GELECEK YOL HARİTASI (ROADMAP)

> **Belge:** Resmi Gelecek Vizyonu ve Sürüm Yol Haritası  
> **Dosya:** `c:\Users\onur_\Desktop\newton-vision\ROADMAP.md`  
> **Mevcut Sürüm:** `v1.1.6` (eğitim/araştırma amaçlı; bilinen açık sorunlar için bkz. `PROGRESS.md` → Açık Sorunlar)  
> **Felsefe:** "Abartılı pazarlama rakamları yerine ölçülmüş mühendislik verileriyle konuşmak; hedefleri açıkça hedef, gerçekleri ise ölçülmüş benchmark olarak sunmak."

---

## 1. STRATEJİK HEDEF: PROJE NEYE DÖNÜŞEBİLİR?

`newton-vision`, 2 GB'lık devasa PyTorch runtime bağımlılıklarını ortadan kaldıran yalın yapısıyla iki temel nişe odaklanmaktadır:

1. **"Edge AI Dünyasının SQLite'ı"**:
   * Nasıl ki SQLite devasa sunuculara ihtiyaç duymadan her telefonda ve IoT cihazında bağımsız çalışıyorsa; `newton-vision` da ağır Python kütüphaneleri olmadan gömülü sistemlerde (Raspberry Pi, ATM, akıllı kamera, medikal donanım) düşük gecikmeli görsel yapay zeka çalıştıran bağımsız bir C++ motoru olmayı hedefler.
2. **"Siber Güvenlik Odaklı Yerli AI Çekirdeği (Cyber-Native AI)"**:
   * EDR (Endpoint Detection & Response) ajanlarının içine doğrudan gömülebilen, PE/ELF ikili dosyalarını ve bellek dökümlerini 2D tensörlere çevirip zararlı yazılımları yerel olarak tespit eden hafif bir tehdit analiz çekirdeği.

---

## 2. MEVCUT ÖLÇÜLMÜŞ BENCHMARK GERÇEKLERİ (VERIFIED BASELINE)

Gelecek hedeflerini konumlandırmadan önce, **şu anki sistemde bizzat ölçülmüş** doğrulanabilir veriler şunlardır:

* **Sayısal Doğruluk (Parite):** PyTorch 2.6.0 referansı ile 4 katmanlı `NewtonVisionCNN` forward çıktısı arasındaki maksimum mutlak fark **`0.00000008`**, kosinüs benzerliği **`%100.000000`**'dir (`apps/pytorch_int8_benchmark.py`). Katman düzeyindeki 85 noktalı matriste en kötü FP32 farkı `2.7e-5`'tir.
* **Bellek Küçülmesi:** INT8 kuantizasyon ile model ağırlık boyutu **1048 KB'tan 262 KB'a** inmiştir (**%74.98 net tasarruf**).
* **LeNet (28x28 Grayscale - MNIST):** Ön işleme (bounding box, ölçekleme, merkezleme) + C++ çıkarımı dahil uçtan uca süre: **`1.99 ms - 2.34 ms` (~427 - 501 FPS)**.
* **4 Katmanlı CNN (32x32 RGB - Batch=1, Salt Çıkarım):** CPU AVX2 üzerinde **`1.28 ms` (779 FPS)** (FP32) ve **`1.54 ms` (649 FPS)** (INT8).
* **Büyük Mimariler (224x224 RGB, batch=1, eval + no_grad, `benchmarks/BENCH_VS_PYTORCH.md`, 2026-10-23):**
  * ResNet-18 (11.7M parametre): ~123 ms, PyTorch CPU'dan ~4.2x yavaş.
  * ViT-Tiny (5.7M parametre): ~61 ms, aynı boyuttaki PyTorch ViT-Tiny'den ~2.5x yavaş (diğer koşularda 3.7x'e kadar). *(Önceki "2.28x daha hızlı" iddiası, PyTorch tarafında ViT-B/16 kullanıldığı için geçersizdir.)*
  * MobileNetV2: ~109 ms, ~4.3x yavaş (C++ optimizasyonundan önce 1565 ms / ~73x; depthwise için özel çekirdek hâlâ yok).
  * Mutlak süreler bu makinede koşudan koşuya belirgin şekilde değişir; oranlar daha kararlıdır.

> [!NOTE]
> Çıkarım hızı; model derinliği, girdi çözünürlüğü (28x28 vs 224x224), batch size ve disk/ön işleme sürelerine doğrudan bağlıdır. Bu nedenle genel bir "600+ FPS motoru" ifadesi yerine her model ailesi kendi koşullarında değerlendirilir.

---

## 3. AŞAMALI SÜRÜM YOL HARİTASI (PHASED ROADMAP)

```
[v1.1.6: Mevcut Temel] 
   └── Autograd, INT8 Kuantizasyon, ViT-Tiny, ResNet18, MobileNetV2, Malware (sentetik) & Vision demoları (unittest: 77/77).
      │
[v1.2: Sertlestirme & Derin Dogrulama (Kisa Vade)]
   └── Autograd Parity Suite, Numerical Gradcheck Genislemesi, INT8 Delta Accuracy, Gradient Clipping.
      │
[v1.3: Düşük Bit & Çapraz Platform (Orta Vade)]
   └── ARM NEON SIMD Desteği, WebAssembly (WASM) Hedefi, FP16 Veri Tipi.
      │
[v1.4: Gerçek Zamanlı Nesne Tespiti (Gelişmiş Vade)]
   └── Tiny-YOLO / Nano-SSD Mimarisi, C++ Vektörize NMS (Non-Maximum Suppression).
      │
[v2.0: Bağımsız C++ SDK & Tehdit İstihbarat Çekirdeği (Kurumsal Seviye)]
   └── Python bağımsız saf C++ header-only SDK, C++ PE/ELF doğrudan ayrıştırıcı, gRPC Edge Dağıtımı.
```

---

## 4. SÜRÜM DETAYLARI VE TEKNİK HEDEFLER

### FAZ 1: v1.2 — SERTLEŞTİRME, DERİN DOĞRULAMA VE TEST KAPSAMI (Hardening & Verification)
*Felsefe: Yeni özellik yığmak yerine mevcut sistemi acımasızca test etmek ve doğrulamak.*
* [ ] **Autograd Primitive Parity Suite:**
  * Her bir temel işlem (`matmul`, `softmax`, `exp`, `log`, `div`, `pow`, `slice`) için PyTorch referansıyla tek tek analitik ve türev paritesi doğrulaması.
* [ ] **Genişletilmiş Sayısal Gradyan Doğrulaması (Finite-Differences Gradcheck):**
  * `tests/test_gradcheck.py` altyapısını Convolution, MultiHeadAttention ve LayerNorm katmanlarına genişletmek.
* [ ] **INT8 Kuantizasyon Doğruluk Kaybı ($\Delta$ Accuracy):**
  * Yalnızca bellek tasarrufunu (%74.98) değil; gerçek sınıflandırma görevlerinde FP32 vs INT8 doğruluk farkını ($\text{Accuracy}_{\text{FP32}} - \text{Accuracy}_{\text{INT8}}$) resmi rapora işlemek.
* [ ] **Gradient Clipping (`clip_grad_norm_`):**
  * Derin modellerde gradyan patlamasını (exploding gradient) önleyen tavan mekanizması.
* [ ] **Learning Rate Schedulers:**
  * `StepLR` ve `CosineAnnealingLR` yakınsama yardımcıları.
* [ ] **Hafif Veri Çoğaltma (Lightweight Augmentations):**
  * Harici kütüphane gerektirmeyen saf NumPy tabanlı `RandomHorizontalFlip`, `RandomCrop`, `Normalize`.
* [ ] **Metrik Paketi:** `Accuracy`, `F1-Score`, `Precision`, `Recall`, `ConfusionMatrix` yardımcıları.

---

### FAZ 2: v1.3 — DONANIM ÇEŞİTLİLİĞİ VE EDGE DAĞITIMI (Orta Vade)
*Modelin farklı işlemci mimarilerinde sıfır bağımlılıkla çalışması:*
* [ ] **ARM NEON SIMD İvmelendirme:**
  * **Hedef:** Raspberry Pi ve Apple Silicon mimarilerinde mevcut skaler CPU döngülerine kıyasla ölçülebilir hızlanma sağlayan vektörize C++ kernel desteği.
* [ ] **WebAssembly (WASM) Derleme:**
  * **Hedef:** Modelin web tarayıcısında (client-side) sunucuya veri iletmeden doğrudan çalıştırılabilmesi.
* [ ] **FP16 (Half-Precision) Desteği:**
  * **Hedef:** FP32'ye kıyasla model ağırlıklarında teorik %50 bellek tasarrufu ve destekleyen donanımlarda throughput artışı.

---

### FAZ 3: v1.4 — GERÇEK ZAMANLI NESNE TESPİTİ (Real-Time Object Detection)
*Sınıflandırmanın ötesine geçerek sınır kutusu (Bounding Box) üretimi:*
* [ ] **Tiny-YOLO / Nano-Detection Head:**
  * Giriş görüntüsündeki nesnelerin koordinatlarını ve sınıflarını tespit edebilme.
* [ ] **C++ Vektörize NMS (Non-Maximum Suppression):**
  * Çakışan aday kutuları hızlıca filtreleyen C++ çekirdeği.
* [ ] **Video Akış Okuyucusu:**
  * `apps/video_stream_detector.py` ile kamera/RTSP akışından kare kare çıkarım boru hattı.

---

### FAZ 4: v2.0 — KURUMSAL C++ MOTORU VE SİBER SAVUNMA PLATFORMU (Uzun Vade)
*Kurumsal seviyede ticarileşebilir büyük ürün vizyonu:*
* [ ] **Header-Only C++ SDK (`newton_vision.hpp`):**
  * Python bağımlılığı olmadan, doğrudan C++ masaüstü uygulamalarına veya EDR yazılımlarına tek `#include` ile bağlanabilme.
* [ ] **C++ Yerel PE/ELF/Mach-O İkili Ayrıştırıcı:**
  * İkili dosyayı Python'a uğramadan doğrudan C++ seviyesinde byte tensörüne döküp zararlı yazılım analizi yapma.
* [ ] **Hafif Mikroservis Dağıtımı (Zero-PyTorch gRPC Container):**
  * PyTorch'un çok gigabaytlık Docker imajları yerine, onlarca megabaytlık ultra-hafif bir inference mikroservisi.

---

## 5. MÜHENDİSLİK İLKELERİ VE ÖLÇÜM DİSİPLİNİ

1. **Hiperbolik Pazarlama Yasağı:** Doğrulanmamış teorik çarpanlar ("4x", "10x") kesin gerçek gibi sunulamaz; "hedeflenen teorik iyileştirme" olarak etiketlenir.
2. **Kapsam Belirtme Zorunluluğu:** Paylaşılan her FPS/gecikme verisi; donanım, model adı, çözünürlük, batch size ve kapsam (salt hesaplama vs uçtan uca I/O) belirtilerek paylaşılır.
3. **Regresyon Koruması:** Yeni özellikler eklenirken [`PROJECT_RULES.md`](file:///c:/Users/onur_/Desktop/newton-vision/PROJECT_RULES.md) kapsamındaki testlerin (`python -m unittest discover tests`, 77 test) kırılmaması şarttır.
