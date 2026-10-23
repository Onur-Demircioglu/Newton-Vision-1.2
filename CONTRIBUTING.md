# 🤝 newton-vision Katkı Sağlama Rehberi (Contributing Guide)

`newton-vision` projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Açık kaynak topluluğunun her türlü katkısını (Bugfix, Dokümantasyon, Test, Performans İyileştirmesi) memnuniyetle karşılıyoruz.

---

## 🛠️ Katkı Adımları (Contribution Steps)

1. **Repository'yi Fork'layın:**
   GitHub üzerinde projeyi kendi hesabınıza fork edin.

2. **Geliştirme Ortamını Kurun:**
   ```bash
   git clone https://github.com/KULLANICI_ADI/newton-vision.git
   cd newton-vision
   pip install -e .[dev]
   ```

3. **Yeni Bir Feature Branch Oluşturun:**
   ```bash
   git checkout -b feature/harika-yeni-ozellik
   ```

4. **Kod ve Test Kuralları:**
   - Eklediğiniz her yeni katman veya fonksiyon için `tests/` klasörü altına unittest eklenmelidir.
   - Tüm birim testler geçmelidir (`python -m unittest discover tests`).
   - C++ değişikliklerinde MSYS2 MinGW / GCC veya MSVC ile derleme hatası olmamalıdır.

5. **Pull Request (PR) Açın:**
   - Yaptığınız değişiklikleri açıklayan kısa ve anlaşılır bir PR başlığı ve açıklaması yazın.

---

## 📏 Kod Satırı & Kalite İlkeleri
- Dürüstlük ve şeffaflık esastır. Performans iddiaları istatistiksel ölçüm ile belgelenmelidir.
- Mevcut kamuya açık (public) API imzalarında kırıcı değişiklik yapılmamalıdır.
