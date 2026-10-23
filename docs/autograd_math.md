# Fluxion Autograd Engine — Matematiksel Türev Derivasyonları

Bu doküman, `newton-vision.fluxion` otomatik türev motorunun arka planındaki matematiksel türev hesabı (Chain Rule), Jakobiyen vektör çarpımları ve evrişim (Convolution) geriye yayılım formüllerini açıklar.

---

## 1. Tam Bağlantılı Katman (Linear / Dense Layer)

Tam bağlantılı katmanda ileri besleme bağıntısı:
$$Y = X W + b$$

Burada:
- $X \in \mathbb{R}^{N \times D_{in}}$ (Giriş matrisi)
- $W \in \mathbb{R}^{D_{in} \times D_{out}}$ (Ağırlık matrisi)
- $b \in \mathbb{R}^{D_{out}}$ (Sapma vektörü)
- $Y \in \mathbb{R}^{N \times D_{out}}$ (Çıktı matrisi)

Geriye yayılım esnasında üst katmandan gelen kayıp türevi $\delta = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times D_{out}}$ olsun. 

### Ağırlık Gradyanı ($\frac{\partial L}{\partial W}$):
$$\frac{\partial L}{\partial W} = X^T \delta \quad \in \mathbb{R}^{D_{in} \times D_{out}}$$

### Giriş Gradyanı ($\frac{\partial L}{\partial X}$):
$$\frac{\partial L}{\partial X} = \delta W^T \quad \in \mathbb{R}^{N \times D_{in}}$$

### Sapma Gradyanı ($\frac{\partial L}{\partial b}$):
$$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \delta_{i, :} \quad \in \mathbb{R}^{D_{out}}$$

---

## 2. 2D Evrişim Katmanı (Conv2d: im2col + GEMM Derivative)

İleri beslemede $X \in \mathbb{R}^{N \times C_{in} \times H \times W}$ tensörü `im2col` dönüşümü ile $X_{col} \in \mathbb{R}^{K \times L}$ matrisine dönüştürülür:
$$K = C_{in} \cdot k_H \cdot k_W, \quad L = H_{out} \cdot W_{out}$$

Yeniden şekillendirilmiş ağırlık matrisi $W_{mat} \in \mathbb{R}^{C_{out} \times K}$ için ileri besleme matris çarpımı:
$$Y_{mat} = W_{mat} X_{col} + b_{mat}$$

Geriye yayılım türevleri ($\delta \in \mathbb{R}^{C_{out} \times L}$):

### Evrişim Ağırlık Gradyanı ($\frac{\partial L}{\partial W}$):
$$\frac{\partial L}{\partial W_{mat}} = \delta X_{col}^T \quad \in \mathbb{R}^{C_{out} \times K}$$

### Evrişim Giriş Gradyanı ($\frac{\partial L}{\partial X}$):
$$\frac{\partial L}{\partial X_{col}} = W_{mat}^T \delta \quad \in \mathbb{R}^{K \times L}$$
$$\frac{\partial L}{\partial X} = \text{col2im}\left(\frac{\partial L}{\partial X_{col}}\right)$$

---

## 3. Aktivasyon Katmanları Türevleri

### ReLU (Rectified Linear Unit):
$$f(x) = \max(0, x) \implies f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$
$$\frac{\partial L}{\partial X} = \delta \odot \mathbb{I}(X > 0)$$

### Sigmoid:
$$\sigma(x) = \frac{1}{1 + e^{-x}} \implies \sigma'(x) = \sigma(x)(1 - \sigma(x))$$
$$\frac{\partial L}{\partial X} = \delta \odot Y \odot (1 - Y)$$
