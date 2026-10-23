# newton-vision — Kalite, Güven & Stabilizasyon Planı (Quality & Trust Roadmap)

> **Vizyon:** Özellik eklemeyi dondurarak projenin geliştiriciye verdiği **güven, belgelendirme, test kapsamı (coverage) ve örnek proje zenginliğini** kurumsal/endüstriyel standartlara ulaştırmak.

---

## 🎯 8 Temel Kalite & Güven Aşaması

### 1. API Stabilizasyonu & Semantik Sürümleme (Semantic Versioning 2.0.0)
- Kamuya açık (Public) API imzalarını dondurmak (`nv.Tensor`, `nv.optics.*`, `nv.models.*`, `nv.jit.*`, `nv.quantization.*`).
- `__all__` listelerini ve modül aktarımlarını eksiksiz tanımlamak.
- Geriye dönük uyumluluğu (Backward Compatibility) garanti altına almak.

---

### 2. Otomatik Benchmark & Performans Yarıştırma Paketi (`benchmarks/`)
- `benchmarks/bench_vs_pytorch.py`:
  - `newton-vision` ile `PyTorch CPU` motorunu ResNet-18, MobileNetV2, ViT-Tiny ve LeNet modellerinde yarıştırmak.
  - Farklı Batch boyutlarında (Batch = 1, 4, 16, 64) Çıkarım Gecikmesi (Latency ms) ve FPS değerlerini otomatik ölçüp Markdown/ASCII tablo raporu üretmek.

---

### 3. Test Kapsamı Raporu (Code Coverage > %90)
- `coverage.py` ve `pytest-cov` ile C++ pybind11 köprüsü ve Python modüllerinin test kapsama oranını ölçmek.
- Test edilmemiş kenar durumları (edge cases) için yeni birim testleri ekleyerek kapsama oranını %90+'ya çıkarmak.

---

### 4. Zengin Örnek Projeler & Öğretici Galerisi (`examples/` & `tutorials/`)
Geliştiricilerin doğrudan kopyalayıp projelerinde kullanabileceği detaylı yorum satırlı 6 uygulama örneği:
1. `01_image_classification_lenet.py`: MNIST Görsel Sınıflandırma ve Başarım Metrikleri.
2. `02_resnet18_pytorch_weights.py`: PyTorch `.pth` Ağırlıklarını Yükleme ve Çıkarım Yapma.
3. `03_vit_transformer_inference.py`: Vision Transformer (ViT) Modeli ile Çıkarım.
4. `04_int8_quantization_demo.py`: INT8 Kuantizasyon ile Model Boyutunu %75 Küçültme.
5. `05_onnx_export_guide.py`: Eğitilen Modeli ONNX Formatına Aktarma.
6. `06_cpp_native_inference.cpp`: Python Bağımlılığı Olmadan Saf C++ Çıkarım Motoru.

---

### 5. Eksiksiz Dokümantasyon & Docstring Temizliği (`docs/`)
- Tüm katmanlar, sınıflar ve fonksiyonlar için Google/NumPy stili eksiksiz docstring'ler yazmak.
- `docs/` klasöründe Sphinx / ReadTheDocs uyumlu HTML dokümantasyon altyapısını oluşturmak.

---

### 6. GitHub Wiki & Kullanım Rehberi (`WIKI.md`)
- Projeyi GitHub üzerinde inceleyen geliştiriciler için detaylı Mimari, Kurulum, Performans ve API Referans kılavuzu hazırlamak.

---

### 7. Katkı Sağlama & Lisans Rehberi (`CONTRIBUTING.md` & `LICENSE`)
- Topluluğun projeye katkı sunabilmesi için `CONTRIBUTING.md` ve MIT Lisans belgelerini düzenlemek.

---

### 8. Proje Güven Sertifikasyonu & Sürüm Notları (`RELEASE_NOTES_v2.0.md`)
- v2.0.0 resmi kararlı sürüm duyurusu ve güvenlik/performans doğrulaması.

---

*Son Güncelleme: 23 Temmuz 2026*
