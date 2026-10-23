# NEWTON-VISION — SIFIR HATA VE GÜVENLİK PROTOKOLÜ (PROJECT RULES)

> **Kritik Derece:** 🔴 EN YÜKSEK (Career-Critical / Zero-Failure Protocol)  
> **Konum:** `c:\Users\onur_\Desktop\newton-vision\PROJECT_RULES.md`  
> **Geçerlilik:** Bu projede kod yazan, değiştiren veya entegrasyon yapan **HER AI Ajanı ve Her Yazılım Mühendisi** bu kurallara istisnasız uymak zorundadır.  
> **Temel İlke:** "Hata çıkarsa işimden olurum" ciddiyetiyle hareket edilecek; varsayıma dayalı, test edilmemiş hiçbir kod tamamlanmış kabul edilmeyecektir.

---

## 1. ANAYASAL ÇALIŞMA KURALLARI

### KURAL 1: TEST KİLİDİ VE REGRESYON YASAĞI (Zero Regression)
* Mevcut 24 birim testin **TAMAMI (%100)** her kod değişikliğinden sonra çalıştırılmalı ve geçmelidir.
* Tek bir test dahi kırmızı (FAIL/ERROR) ise yapılan değişiklik **asla onaylanamaz ve bitmiş kabul edilemez**.
* Komut: `python -m unittest discover tests`

### KURAL 2: KANITLANMAMIŞ İLERLEME YASAKTIR (Proof-First Engineering)
* "Kod yazıldı, muhtemelen çalışır", "mantıken doğru görünüyor" gibi yaklaşımlar **kesinlikle yasaktır**.
* Bir özelliğin tamamlandığı ancak ve ancak terminalde canlı çalıştırılıp çıktısı, gradyan normu veya sayısal değeri konsolda bizzat görüldükten sonra deklare edilebilir.

### KURAL 3: SAVUNMACI PROGRAMLAMA (Defensive Guards & Crash Barrier)
* Kullanıcıdan veya dış dünyadan gelen hiçbir girdi (görsel boyutu, tensör şekli, dosya yolu, ikili veri) doğrudan C++ çekirdeğine iletilemez.
* Tüm public katmanlar (`Conv2d`, `Linear`, `ImageFolder`, `predict_image` vb.) giriş verisini savunmacı muhafızlarla (Shape & Dimension Guards) doğrulamak zorundadır.
* Geçersiz bir girdide program çökmeyecek (SIGSEGV veya unhandled crash olmayacak), temiz ve açıklayıcı bir `ValueError` fırlatacaktır.

### KURAL 4: SİBER GÜVENLİK VE GÜVENLİ SERİLEŞTİRME (Strict Deserialization)
* Proje genelinde `pickle`, `pickle.load` veya `eval()` kullanımı **KATI ŞEKİLDE YASAKTIR**.
* Ağırlık yükleme fonksiyonlarında (`np.load`) `allow_pickle=False` parametresi zorunludur.
* Model serileştirme yalnızca saf ikili format (`.nv.bin`) veya güvenli NumPy array (`.nv`) üzerinden yapılabilir.

### KURAL 5: MİMARİ DİSİPLİN (Module & Parameter Contract)
* Yeni eklenecek her katman `newton_vision.module.Module` sınıfından türetilmek zorundadır.
* Eğitilebilir tensörler `Parameter` olarak işaretlenmeli ve `self.register_parameter` ile hiyerarşiye bağlanmalıdır.
* Her katman `self.training` durumunu (`model.train()` ve `model.eval()`) doğru şekilde yönetmelidir (Örn: `Dropout`, `BatchNorm2d`).

### KURAL 6: PYTORCH PARİTE STANDARDI
* Temel tensör ve katman çıktılarında PyTorch referansı ile aradaki fark toleransı maksimum `1e-6` olmalıdır.
* PyTorch'tan aktarılan ağırlıklarda transpoz (`(out, in) -> (in, out)`) uyumsuzlukları `weights.py` tarafından otomatik ve sessizce çözülmelidir.

### KURAL 7: CANLI DEMO DOKUNULMAZLIĞI (Demo Immunity)
* `apps/` klasöründeki 3 ana gösterim aracı ve `assets/` klasöründeki test verileri dokunulmazdır:
  1. `apps/image_predictor.py` (Görsel tahmin & ASCII sunumu)
  2. `apps/malware_classifier.py` (PE binary malware CNN analizi)
  3. `apps/pytorch_int8_benchmark.py` (PyTorch parite & INT8 benchmark)
* Bu uygulamaların her an tek tıkla hatasız çalışabilir durumda kalması zorunludur.

---

## 2. DEĞİŞİKLİK YAPILIRKEN ZORUNLU KONTROL LİSTESİ (CHECKLIST)

Bir dosya değiştirildiğinde geliştiricinin / AI'ın izlemesi gereken sıra:

1. [ ] İlgili modülün [`AI_MEMORY.md`](file:///c:/Users/onur_/Desktop/newton-vision/AI_MEMORY.md) dökümanındaki geçmişine bakıldı mı?
2. [ ] Yeni kod yazılırken `Module` / `Parameter` kurallarına uyuldu mu?
3. [ ] Giriş parametrelerine savunmacı sınır kontrolü (guard) konuldu mu?
4. [ ] Değişiklik için yeni bir unit test eklendi mi?
5. [ ] `python -m unittest discover tests` çalıştırılıp 24/24 yeşil görüldü mü?
6. [ ] `apps/` demoları çalıştırılıp canlı teyit alındı mı?
7. [ ] [`AI_MEMORY.md`](file:///c:/Users/onur_/Desktop/newton-vision/AI_MEMORY.md) güncellendi mi?

---

## 3. ASLA YAPILMAYACAKLAR LİSTESİ (RED LINES)

❌ **YAPMA:** Kullanıcı test etmeden "hallettim" deme.  
❌ **YAPMA:** C++ tarafında boyut kontrolü yapmadan belleğe pointer erişimi verme.  
❌ **YAPMA:** Geriye türevi (backward) yazılmamış yarım tensör fonksiyonu bırakma.  
❌ **YAPMA:** `try/except` bloğunda hatayı yutup sessizce geçme (`pass` koyma).  
❌ **YAPMA:** Mevcut testleri devre dışı bırakarak veya silerek başarı taklidi yapma.
