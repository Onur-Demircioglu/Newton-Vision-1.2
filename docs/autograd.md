# 🧮 Otomatik Türev Motoru Matematiği (Reverse-Mode Autograd)

`newton-vision.fluxion`, ters mod (reverse-mode) otomatik türev türetimi kullanarak karmaşık sinir ağı grafiklerinin türevlerini otomatik hesaplar.

---

## 📐 Matematiksel Formulasyon

Bir $y = f(x_1, x_2, \dots, x_n)$ fonksiyonu için, geriye yayılım adımında zincir kuralı (Chain Rule) işletilir:

$$\frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}$$

---

## 🔄 Ters Topolojik Sıralama (Topological Sort)

1. İleri geçişte (forward pass) tensör bağımlılıkları düğüm grafına eklenir (`_children`).
2. `tensor.backward()` çağrıldığında, türev grafı ters topolojik sıralama algoritması ile düğümlere ayrıştırılır:
3. Her düğümün türev fonksiyonu (`_backward()`) sırayla çalıştırılarak gradyan biriktirilir (`grad += local_grad`).
