# NEWTON-VISION — MATEMATİKSEL TEMELLER VE TÜREV DERİVASYONLARI KİTABI (MATHEMATICS OF NEWTON-VISION)

> **Belge Tipi:** Tam Matematiksel Şartname ve Formülasyon Kılavuzu  
> **Konum:** `c:\Users\onur_\Desktop\newton-vision\MATHEMATICS.md`  
> **Sürüm:** `v1.1.6 (Specification & Code Parity Engine)`  
> **Kapsam:** Çerçevede (Framework) uygulanan tensör cebiri, otomatik türev (Autograd), evrişim (Convolution), normalizasyon, dikkat mekanizması, optimizatörler ve INT8 kuantizasyon matematiğinin kod ile birebir tutarlı eksiksiz dökümü.

---

## SİMGELER VE GÖSTERİMLER (NOMENCLATURE)

Notasyonel çakışmaları ve anlam belirsizliklerini önlemek için kullanılan standart simgeler tablosu:

| Sembol | Tanım | Tipik Değer / Tanım Kümesi |
| :--- | :--- | :--- |
| $\delta$ veya $\nabla_Y L$ | Üst katmandan gelen geriye yayılım gradyanı ($\frac{\partial L}{\partial Y}$) | $\mathbb{R}^{\text{shape}(Y)}$ |
| $G_{\text{group}}$ | Evrişim katmanı grup sayısı (groups) | $G_{\text{group}} \ge 1$ ($C_{\text{in}}$ ve $C_{\text{out}}$'u tam böler) |
| $D$ veya $d_{\text{model}}$ | Model / Özellik gizli boyutu (Feature / Embedding dimension) | $\mathbb{N}^+$ (örneğin 64, 128, 768) |
| $d_{\text{dil}}$ | Evrişim genişletme katsayısı (Dilation rate) | $\mathbb{N}^+$ (varsayılan: 1) |
| $S_{\text{stride}}$ | Evrişim veya havuzlama adım boyutu (Stride) | $\mathbb{N}^+$ (varsayılan: 1) |
| $S_{\text{seq}}$ | Dizi uzunluğu (Sequence length / Token count) | $\mathbb{N}^+$ |
| $s_{\text{scale}}$ | Kuantizasyon ölçek çarpanı (Quantization scale factor) | $\mathbb{R}^+$ ($s_{\text{scale}} = \max(\|X\|) / 127.0$) |
| $m_{\text{mask}}$ | Dropout ikili seçim maskesi | $\{0, 1\}^{\text{shape}(X)}$ |
| $m_{\text{mom}}$ | BatchNorm hareketli ortalama momentum katsayısı | $m_{\text{mom}} \in [0, 1]$ (varsayılan: 0.1) |
| $m_t$ | Adam optimizatörü 1. moment (hareketli gradyan ortalaması) | $\mathbb{R}^{\text{shape}(\theta)}$ |
| $v_t$ | Adam optimizatörü 2. moment (hareketli gradyan karesi ortalaması) | $\mathbb{R}^{\text{shape}(\theta)}$ |
| $\mu_{\text{sgd}}$ | SGD optimizatörü momentum katsayısı | $\mu_{\text{sgd}} \ge 0$ |
| $\mu$ | İstatistiksel örneklem aritmetik ortalaması (Mean) | $\mathbb{R}$ veya $\mathbb{R}^C$ |
| $\epsilon_{\text{norm}}$ | Normalizasyon payda kararlılık sabiti (Epsilon) | $10^{-5}$ |
| $\epsilon_{\text{adam}}$ | Adam payda sıfıra bölme koruma sabiti | $10^{-8}$ |
| $\epsilon_{\text{step}}$ | Sayısal gradcheck merkezi fark adım boyutu | $10^{-3} \dots 10^{-4}$ (FP32 için optimum) |

---

## İÇİNDEKİLER

1. [Tensör Uzayı, Bellek Modeli ve Strides](#1-tensör-uzayı-bellek-modeli-ve-strides)
2. [Ters Mod Otomatik Türev Motoru (Reverse-Mode Autograd Calculus)](#2-ters-mod-otomatik-türev-motoru-reverse-mode-autograd-calculus)
3. [Temel Tensör Operatörleri, Şekil Dönüşümleri ve İndirgemeler](#3-temel-tensör-operatörleri-şekil-dönüşümleri-ve-indirgemeler)
4. [Tam Bağlantılı Katman (Linear / Dense Layer)](#4-tam-bağlantılı-katman-linear--dense-layer)
5. [2D Evrişim Katmanı (Conv2d: Gruplu im2col + GEMM Calculus)](#5-2d-evrişim-katmanı-conv2d-gruplu-im2col--gemm-calculus)
6. [Havuzlama Katmanları (MaxPool2d ve AvgPool2d)](#6-havuzlama-katmanları-maxpool2d-ve-avgpool2d)
7. [Aktivasyon Fonksiyonları ve Kapalı Form Türevleri](#7-aktivasyon-fonksiyonları-ve-kapalı-form-türevleri)
8. [Normalizasyon Katmanları (BatchNorm2d ve LayerNorm)](#8-normalizasyon-katmanları-batchnorm2d-ve-layernorm)
9. [Düzenlileştirme: Ters Ölçeklemeli Dropout (Inverted Dropout)](#9-düzenlileştirme-ters-ölçeklemeli-dropout-inverted-dropout)
10. [Çok Başlıklı Öz-Dikkat Mekanizması (Multi-Head Self-Attention - MHSA)](#10-çok-başlıklı-öz-dikkat-mekanizması-multi-head-self-attention---mhsa)
11. [Kayıp Fonksiyonları ve Sayısal Kararlılık (Loss Calculus)](#11-kayıp-fonksiyonları-ve-sayısal-kararlılık-loss-calculus)
12. [Optimizasyon Algoritmaları (SGD Momentum ve Bağlı L2 Adam)](#12-optimizasyon-algoritmaları-sgd-momentum-ve-bağlı-l2-adam)
13. [INT8 Kuantizasyon ve Afin Haritalama Matematiği](#13-int8-kuantizasyon-ve-afin-haritalama-matematiği)
14. [Sayısal Gradyan Doğrulama ve Gradcheck Metodolojisi](#14-sayısal-gradyan-doğrulama-ve-gradcheck-metodolojisi)

---

## 1. TENSÖR UZAYI, BELLEK MODELİ VE STRIDES

Bir tensör $T \in \mathbb{R}^{d_0 \times d_1 \times \dots \times d_{n-1}}$, $n$-boyutlu Kartezyen uzayda ayrık bir veri bloğudur.

### 1.1. C-Contiguous Bellek Haritası
Tensör verileri bilgisayar belleğinde 1D düz dizi (`flat buffer`) olarak saklanır. $n$-boyutlu bir $(i_0, i_1, \dots, i_{n-1})$ koordinatının bellekteki eleman ofseti adım vektörü $S \in \mathbb{Z}^n$ ile hesaplanır:

$$\text{Offset}(i_0, i_1, \dots, i_{n-1}) = \sum_{k=0}^{n-1} i_k \cdot S_k$$

Satır-öncelikli (Row-major / C-contiguous) düzende adım vektörü $S_k$:
$$S_k = \prod_{j=k+1}^{n-1} d_j, \quad S_{n-1} = 1 \quad (\text{eleman cinsinden})$$

> **Strided Views ve In-Place Kısıtları:** Transpose ve permute operasyonlarında veriler fiziksel olarak kopyalanmadığında (strided view), $S_k$ genel adım değerleri alır. Broadcast görünümlerinde genişletilen eksenin stride değeri $0$'dır; bu nedenle stride $0$ olan tensör görünümlerine doğrudan in-place yazma bellekte çakışma yaratacağından yasaklanmalıdır.

---

## 2. TERS MOD OTOMATİK TÜREV MOTORU (REVERSE-MODE AUTOGRAD CALCULUS)

Hesaplama grafı (Directed Acyclic Graph - DAG) üzerinde zincir kuralının (Chain Rule) analitik ve dinamik yürütümüdür.

### 2.1. Yönlendirilmiş Döngüsüz Çizge (DAG) ve Topolojik Sıralama
Her tensör bir düğüm ($v \in V$), her tensör işlemi ise yönlendirilmiş bir kenardır ($e \in E$).
İleri beslemede skaler kayıp $L = f(v_1, v_2, \dots, v_m)$ üretilir.

Geriye türev (backward pass) için düğümler topolojik olarak sıralanır:
$$\text{Topo}(V) = [v_{\text{root}}, \dots, v_{\text{leaf}}]$$
Böylece bir düğümün gradyanı hesaplanmadan önce, ondan türeyen tüm çocuk düğümlerin gradyan katkıları toplanmış olur.

### 2.2. Vektör-Jakobiyen Çarpımı (Vector-Jacobian Product - VJP)
Bir $f: \mathbb{R}^n \to \mathbb{R}^m$ fonksiyonu için üst katmandan gelen kayıp türevi $\delta = \nabla_y L \in \mathbb{R}^{1 \times m}$ vektörü ile VJP doğrudan hesaplanır:
$$\nabla_x L = \delta \cdot J = \sum_{i=1}^m \delta_i \frac{\partial y_i}{\partial x_j}$$

### 2.3. Boyut Geri Yayılımı ve Yayma (Unbroadcasting Calculus)
NumPy / PyTorch tarzı boyut yayma (broadcasting) yapıldığında:
$$\text{Eğer } Y = \text{Broadcast}(X, \text{shape}_Y) \implies \frac{\partial L}{\partial X} = \sum_{\text{yayılan eksenler}} \frac{\partial L}{\partial Y}$$

1. Eğer $X$ ve $Y$ rank farkına sahipse ($Y.\text{ndim} > X.\text{ndim}$), baştaki fazla boyutlar toplanarak eritilir:
   $$\nabla_X = \sum_{k=0}^{\text{ndim}_Y - \text{ndim}_X - 1} \nabla_Y$$
2. Eğer $X$'in herhangi bir boyutu 1 ise ve $Y$'de bu boyut $>1$ ise:
   $$\nabla_X = \sum_{k} \nabla_Y, \quad \text{axis } k \text{ boyunca (keepdims=True)}$$
3. Sonuç tensörü $X.\text{shape}$ boyutuna yeniden şekillendirilir (`reshape`).

### 2.4. Topolojik Fan-Out ve Artık Bağlantı (Residual) Toplama Kuralı
Bir tensör $X$, hesaplama grafında birden fazla operasyonun girdisi olduğunda (örneğin artık bağlantı $Y = X + F(X)$ veya MHSA'da $X \to Q, K, V$ dallanması):
$$\frac{\partial L}{\partial X} = \sum_{k \in \text{Tüketiciler}} \left(\frac{\partial L}{\partial X}\right)_k$$
newton-vision autograd motorunda `_add_grad` metodu çağrıldığında gradyanlar üzerine eklenir (`+=`). Bu kural bileşik ağlarda parametre ve aktivasyon gradyanlarının korunmasını garanti eder.

### 2.5. `no_grad()` Bağlamı ve In-Place Değişim Riskleri
* `no_grad()` bloğu içerisinde `_GRAD_ENABLED = False` yapılır; hiçbir tensör işlemi graf düğümü veya `_children` referansı oluşturmaz. Çıkarım (inference) sırasında bellek ayak izi ve hesaplama maliyeti asgariye iner.
* **In-Place Bellek Uyarısı:** İleri geçişte sigmoid, tanh gibi aktivasyonlar geriye yayılım için kendi çıktıları olan $Y$'yi önbelleğe alır. Kullanıcının `tensor.data[...] = ...` şeklinde in-place mutasyon yapması önbellekteki değerleri bozacağından autograd türev doğruluğunu sakatlar.

---

## 3. TEMEL TENSÖR OPERATÖRLERİ, ŞEKİL DÖNÜŞÜMLERİ VE İNDİRGEMELER

Tensör $A$ ve $B$, çıktı $Z$ ve üst gradyan $\delta = \frac{\partial L}{\partial Z}$ olsun.

### 3.1. Eleman Düzeyinde Temel Operatörler
* **Toplama ($Z = A + B$):**
  $$\nabla_A L = \delta, \quad \nabla_B L = \delta$$
  *(Şekil farkı varsa her iki koldan da Madde 2.3'e göre unbroadcast edilir).*
* **Çıkarma ($Z = A - B$):**
  $$\nabla_A L = \delta, \quad \nabla_B L = -\delta$$
* **Hadamard Çarpımı ($Z = A \odot B$):**
  $$\nabla_A L = \delta \odot B, \quad \nabla_B L = \delta \odot A$$
* **Bölme ($Z = \frac{A}{B}$):**
  $$\nabla_A L = \frac{\delta}{B}, \quad \nabla_B L = -\delta \odot \frac{A}{B^2}$$
  > **Kenar Durumu:** $B_i = 0$ noktalarında sonuç $\pm\infty$ veya NaN olur. Kod tabanında paydada sıfır kontrolü gereklidir.
* **Üs Alma ($Z = A^p$, $p \in \mathbb{R}$):**
  $$\nabla_A L = \delta \odot \left(p \cdot A^{p-1}\right)$$
  > **Kenar Durumu:** $A \le 0$ ve tam sayı olmayan $p$ durumunda reel tanım kümesi dışına çıkılır. $A = 0$ ve $p < 1$ durumunda $0^{p-1}$ sonsuza gider.

### 3.2. Matris Çarpımı (Batch Matrix Multiplication)
$A \in \mathbb{R}^{B \times M \times K}$, $B \in \mathbb{R}^{B \times K \times N}$, $Z = A @ B \in \mathbb{R}^{B \times M \times N}$:
$$\nabla_A L = \delta @ \text{swapaxes}(B, -1, -2)$$
$$\nabla_B L = \text{swapaxes}(A, -1, -2) @ \delta$$
Eğer $B$ batch boyutuna sahip değilse ($B \in \mathbb{R}^{K \times N}$), $\nabla_B L$ batch eksenleri üzerinden toplanır:
$$\nabla_B L = \sum_{\text{batch}} \text{swapaxes}(A, -1, -2) @ \delta = A_{\text{flat}}^T \cdot \delta_{\text{flat}}$$

### 3.3. Şekil Dönüşümleri (Shape Transformations)
* **Reshape:** $Z = X.\text{reshape}(*s) \implies \nabla_X L = \delta.\text{reshape}(X.\text{shape})$
* **Transpose / Permute:** $Z = X.\text{transpose}(*axes) \implies \nabla_X L = \delta.\text{transpose}(\text{argsort}(axes))$
* **Concatenate:** $Z = \text{Concat}([T_1, \dots, T_k], \text{axis}=a)$
  $$\nabla_{T_i} L = \delta[\dots, \text{start}_i:\text{end}_i, \dots]$$

### 3.4. İndirgemeler (Reductions)
* **Sum:** $Z = \sum_{\text{axes}} X \implies \nabla_X L = \text{expand\_dims}(\delta, \text{axes}) \cdot \mathbf{1}_{\text{shape}(X)}$
* **Mean:** $Z = \frac{1}{K} \sum_{\text{axes}} X \implies \nabla_X L = \text{expand\_dims}(\delta, \text{axes}) \cdot \frac{1}{K} \cdot \mathbf{1}_{\text{shape}(X)}$  
  burada $K = \prod_{a \in \text{axes}} d_a$.
* **Max:** $Z = \max_{\text{axes}} X$
  $$M = \mathbb{I}(X == \text{expand\_dims}(Z, \text{axes})), \quad C = \sum_{\text{axes}} M$$
  $$\nabla_X L = M \odot \frac{\text{expand\_dims}(\delta, \text{axes})}{C}$$
  *(Eşit en büyük değerler arasında gradyan eşit paylaştırılır).*

### 3.5. Sıfır-Kopya Dilimleme ve Saçılım Toplamı (Scatter-Add)
* **İleri:** $Z = X[\text{idx}]$
* **Geri:** Gradyan orijinal koordinatlara saçılır (`np.add.at`):
  $$\nabla_X L = \mathbf{0}_{\text{shape}(X)}, \quad \nabla_X L[\text{idx}] \mathrel{+}= \delta$$
  *(Embedding katmanı aramaları ve indeksli tensör erişimleri bu mekanizma ile türetilir).*

### 3.6. Elemanter Matematik Fonksiyonları
* **Exp:** $Z = \exp(X) \implies \nabla_X L = \delta \odot Z$
* **Log:** $Z = \ln(X) \implies \nabla_X L = \delta / X \quad (X > 0)$
* **Sqrt:** $Z = \sqrt{X} \implies \nabla_X L = \delta \odot \frac{0.5}{Z} \quad (X > 0)$

---

## 4. TAM BAĞLANTILI KATMAN (LINEAR / DENSE LAYER)

### 4.1. İleri Besleme
$$Y = X W + b$$
* $X \in \mathbb{R}^{N \times D_{\text{in}}}$ (veya çok boyutlu $X \in \mathbb{R}^{B \times S_{\text{seq}} \times D_{\text{in}}}$)
* $W \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ *(newton-vision $y = xW + b$ formatı)*
* $b \in \mathbb{R}^{D_{\text{out}}}$
* $Y \in \mathbb{R}^{N \times D_{\text{out}}}$

### 4.2. Geriye Yayılım Formülleri ($\delta = \frac{\partial L}{\partial Y}$)
2D girdi durumunda:
$$\frac{\partial L}{\partial W} = X^T \cdot \delta \quad \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$$
$$\frac{\partial L}{\partial X} = \delta \cdot W^T \quad \in \mathbb{R}^{N \times D_{\text{in}}}$$
$$\frac{\partial L}{\partial b} = \sum_{i=1}^N \delta_{i, :} \quad \in \mathbb{R}^{D_{\text{out}}}$$

3D girdi durumunda ($X \in \mathbb{R}^{B \times S_{\text{seq}} \times D_{\text{in}}}$, $\delta \in \mathbb{R}^{B \times S_{\text{seq}} \times D_{\text{out}}}$):
$$\frac{\partial L}{\partial W} = X_{\text{flat}}^T \cdot \delta_{\text{flat}}, \quad X_{\text{flat}} \in \mathbb{R}^{(B \cdot S_{\text{seq}}) \times D_{\text{in}}}, \quad \delta_{\text{flat}} \in \mathbb{R}^{(B \cdot S_{\text{seq}}) \times D_{\text{out}}}$$
$$\frac{\partial L}{\partial X} = \delta @ W^T \quad \in \mathbb{R}^{B \times S_{\text{seq}} \times D_{\text{in}}}$$
$$\frac{\partial L}{\partial b} = \sum_{b=1}^B \sum_{s=1}^{S_{\text{seq}}} \delta(b, s, :) \quad \in \mathbb{R}^{D_{\text{out}}}$$

---

## 5. 2D EVRİŞİM KATMANI (CONV2D: GRUPLU IM2COL + GEMM CALCULUS)

### 5.1. Ayrık Evrişim / Çapraz Korelasyon Tanımı
Giriş $X \in \mathbb{R}^{N \times C_{\text{in}} \times H \times W}$, Çekirdek $W \in \mathbb{R}^{C_{\text{out}} \times (C_{\text{in}}/G_{\text{group}}) \times k_H \times k_W}$, Grup sayısı $G_{\text{group}}$, Adım $S_{\text{stride}}$, Dolgu $P$, Genişletme $d_{\text{dil}}$:

Çıktı boyutu:
$$H_{\text{out}} = \left\lfloor \frac{H + 2P - d_{\text{dil}}(k_H - 1) - 1}{S_{\text{stride}}} \right\rfloor + 1$$
$$W_{\text{out}} = \left\lfloor \frac{W + 2P - d_{\text{dil}}(k_W - 1) - 1}{S_{\text{stride}}} \right\rfloor + 1$$

Her $c_{\text{out}} \in [0, C_{\text{out}}-1]$ çıktısı için analitik forward formülü:
$$Y(n, c_{\text{out}}, h, w) = b(c_{\text{out}}) + \sum_{c=0}^{\frac{C_{\text{in}}}{G_{\text{group}}}-1} \sum_{kh=0}^{k_H-1} \sum_{kw=0}^{k_W-1} X\left(n, g \cdot \frac{C_{\text{in}}}{G_{\text{group}}} + c, h \cdot S_{\text{stride}} + kh \cdot d_{\text{dil}} - P, w \cdot S_{\text{stride}} + kw \cdot d_{\text{dil}} - P\right) \cdot W(c_{\text{out}}, c, kh, kw)$$
burada ilgili kanal grubu indeksi $g = \left\lfloor \frac{c_{\text{out}}}{C_{\text{out}} / G_{\text{group}}} \right\rfloor$'dir.

### 5.2. Grup Bazlı `im2col` Matrisleştirme İndirgemesi
newton-vision C++ performans çekirdeğinde (`cpp/layers.cpp`), hesaplama her batch $n \in [0, N-1]$ ve her grup $g \in [0, G_{\text{group}}-1]$ için bağımsız GEMM bloklarına ayrıştırılır:

1. Giriş alt bloğu $X_{n, g} \in \mathbb{R}^{\frac{C_{\text{in}}}{G_{\text{group}}} \times H \times W}$ açılarak:
   $$X_{\text{col}}^{(g)} \in \mathbb{R}^{\left(\frac{C_{\text{in}}}{G_{\text{group}}} \cdot k_H \cdot k_W\right) \times (H_{\text{out}} \cdot W_{\text{out}})}$$
2. Ağırlık alt matrisi $W_{\text{mat}}^{(g)}$:
   $$W_{\text{mat}}^{(g)} \in \mathbb{R}^{\frac{C_{\text{out}}}{G_{\text{group}}} \times \left(\frac{C_{\text{in}}}{G_{\text{group}}} \cdot k_H \cdot k_W\right)}$$
3. Grup GEMM çıktısı:
   $$Y_{n, g} = W_{\text{mat}}^{(g)} \cdot X_{\text{col}}^{(g)} + b^{(g)} \quad \in \mathbb{R}^{\frac{C_{\text{out}}}{G_{\text{group}}} \times (H_{\text{out}} \cdot W_{\text{out}})}$$
4. **Özel Durumlar:**
   * $G_{\text{group}} = 1$: Standart tam evrişim (Standard Convolution).
   * $G_{\text{group}} = C_{\text{in}} = C_{\text{out}}$: Derinlikli ayrılabilir evrişim (Depthwise Separable Convolution).

### 5.3. Evrişim Geriye Yayılımı ($\delta = \frac{\partial L}{\partial Y}$)
Her $n \in [0, N-1]$ ve her grup $g \in [0, G_{\text{group}}-1]$ için:
* **Filtre Gradyanı:**
  $$\frac{\partial L}{\partial W_{\text{mat}}^{(g)}} = \sum_{n=0}^{N-1} \delta_{n, g} \cdot \left(X_{\text{col}}^{(g)}\right)^T \quad \in \mathbb{R}^{\frac{C_{\text{out}}}{G_{\text{group}}} \times \left(\frac{C_{\text{in}}}{G_{\text{group}}} \cdot k_H \cdot k_W\right)}$$
* **Giriş Tensör Gradyanı (`col2im`):**
  $$\delta_{\text{col}}^{(g)} = \left(W_{\text{mat}}^{(g)}\right)^T \cdot \delta_{n, g}$$
  $$\left(\frac{\partial L}{\partial X_{n, g}}\right) = \text{col2im}\left(\delta_{\text{col}}^{(g)}\right)$$
  *(Örtüşen pencerelerin gradyanları orijinal koordinatlara atomik veya OpenMP korumalı olarak `+=` ile toplanır).*
* **Sapma Gradyanı:**
  $$\frac{\partial L}{\partial b(c_{\text{out}})} = \sum_{n=0}^{N-1} \sum_{h=0}^{H_{\text{out}}-1} \sum_{w=0}^{W_{\text{out}}-1} \delta(n, c_{\text{out}}, h, w)$$

---

## 6. HAVUZLAMA KATMANLARI (MAXPOOL2D VE AVGPOOL2D)

### 6.1. MaxPool2d (Maksimum Havuzlama)
* **İleri Besleme:**
  $$Y(n, c, oh, ow) = \max_{kh \in [0, k_H-1], kw \in [0, k_W-1]} X(n, c, oh \cdot S_{\text{stride}} + kh, ow \cdot S_{\text{stride}} + kw)$$
* **Eşitlik Çözüm Kuralı (Tie-Break Policy):**
  C++ çekirdeğinde (`layers.cpp`) `if (v > max_val)` katı büyüktür sınaması uygulandığından satır-öncelikli (row-major: $kh$ dış, $kw$ iç döngü) taramada rastlanan **ilk indeks** argmax olarak seçilir. Bu kural PyTorch standart CPU davranışı ile birebir aynıdır.
* **Dolgu Değeri (Padding):** Negatif girişlerde yanlış maksimum seçimini önlemek için dolgu değeri $-\infty$ kabul edilir.
* **Geri Yayılım (Argmax Routing):**
  İleri geçişte kaydedilen maske $M(n, c, oh, ow) = (ih^*, iw^*)$ kullanılarak gradyan yalnızca kazanan koordinata aktarılır:
  $$\frac{\partial L}{\partial X(n, c, h, w)} = \sum_{(oh, ow)} \delta(n, c, oh, ow) \cdot \mathbb{I}(M(n, c, oh, ow) = (h, w))$$
  *(Örtüşen pencerelerde stride < kernel_size ise gradyanlar birikir).*

### 6.2. AvgPool2d (Ortalama Havuzlama)
* **İleri Besleme:**
  $$Y(n, c, oh, ow) = \frac{1}{k_H \cdot k_W} \sum_{kh=0}^{k_H-1} \sum_{kw=0}^{k_W-1} X(n, c, oh \cdot S_{\text{stride}} + kh, ow \cdot S_{\text{stride}} + kw)$$
* **Bölen Kuralı ve Kapsam:**
  newton-vision mimarisinde havuzlama dolgusuz ($P=0$, `ceil_mode=False`) uygulanır. Bölen sabit alan $k_H \cdot k_W$'dir (PyTorch `count_include_pad=True` moduyla özdeş).
* **Geri Yayılım:**
  $$\frac{\partial L}{\partial X(n, c, h, w)} = \sum_{(oh, ow) \text{ kapsayan}} \frac{\delta(n, c, oh, ow)}{k_H \cdot k_W}$$

---

## 7. AKTİVASYON FONKSİYONLARI VE KAPALI FORM TÜREVLERİ

$\delta = \frac{\partial L}{\partial Y}$ üst gradyan olmak üzere:

### 7.1. ReLU
$$Y = \max(0, X), \quad \nabla_X L = \delta \odot \mathbb{I}(X > 0)$$

### 7.2. LeakyReLU ($\alpha = 0.01$)
$$Y = \max(\alpha X, X), \quad \nabla_X L = \delta \odot \left(\mathbb{I}(X > 0) + \alpha \cdot \mathbb{I}(X \le 0)\right)$$

### 7.3. Sigmoid
$$Y = \sigma(X) = \frac{1}{1 + e^{-X}}$$
Sayısal kararlı hesaplama: $X \ge 0$ için $\frac{1}{1 + e^{-X}}$, $X < 0$ için $\frac{e^X}{1 + e^X}$. $X = \pm 1000$ gibi uç noktalarda $0$ veya $1$ değerine düzgün doyuma ulaşır.
$$\sigma'(X) = Y \odot (1 - Y) \implies \nabla_X L = \delta \odot Y \odot (1 - Y)$$

### 7.4. Tanh
$$Y = \tanh(X) \implies \nabla_X L = \delta \odot (1 - Y^2)$$

### 7.5. GELU (Gaussian Error Linear Unit)
newton-vision, Hendrycks & Gimpel Tanh yaklaşımını kullanır:
$$c = \sqrt{\frac{2}{\pi}} \approx 0.79788456$$
$$u(X) = c \cdot \left(X + 0.044715\,X^3\right)$$
$$t = \tanh(u(X))$$
$$Y = 0.5\,X \cdot (1 + t)$$

Türev, ileri geçişte işletilen **fonksiyonun tam analitik türevidir**:
$$u'(X) = c \cdot \left(1 + 3 \times 0.044715\,X^2\right)$$
$$\frac{dY}{dX} = 0.5\,(1 + t) + 0.5\,X \cdot (1 - t^2) \cdot u'(X)$$
$$\nabla_X L = \delta \odot \frac{dY}{dX}$$

> **Parite Notu:** PyTorch varsayılanı tam Erf fonksiyonudur ($\Phi(X) = \frac{1}{2}(1 + \text{erf}(X/\sqrt{2}))$). Bu iki formülasyon arasında $\sim 10^{-4}$ mertebesinde yaklaşım farkı vardır. newton-vision bu nedenle PyTorch'un `approximate="tanh"` modu ile birebir sayısal parite gösterir.

### 7.6. Softmax ve Vektör Jakobiyen Çarpımı
Taşma korumalı form: $\hat{z} = z - \max(z, \text{axis}=-1, \text{keepdims}=\text{True})$.
$$S_i = \frac{e^{\hat{z}_i}}{\sum_j e^{\hat{z}_j}}$$

Jakobiyen vektör çarpımı:
$$\nabla_z L = S \odot \left( \delta - \sum_k (\delta_k \cdot S_k) \mathbf{1} \right) = S \odot (\delta - \text{sum}(\delta \odot S, \text{axis}=-1, \text{keepdims}=\text{True}))$$

> **Kenar Durumu (Tümüyle $-\infty$ Olan Satır):** Maskeli dikkat işlemlerinde satırdaki tüm değerler $-\infty$ ise $e^{-\infty}=0 \implies \sum=0$ olur ve $\frac{0}{0}$ belirsizliğiyle NaN üretilir. Giriş tensörünün her satırında en az bir sonlu değer bulunmalıdır.

---

## 8. NORMALİZASYON KATMANLARI (BATCHNORM2D VE LAYERNORM)

### 8.1. BatchNorm2d
Giriş $X \in \mathbb{R}^{N \times C \times H \times W}$. Kanal başına örneklem sayısı $M = N \cdot H \cdot W$. $M \le 1$ durumunda varyans tanımsız olacağından eğitim modunda istisna fırlatılır.

#### İleri Besleme:
* **Eğitim Modu:**
  $$\mu_c = \frac{1}{M} \sum_{n, h, w} X, \quad \sigma_c^2 = \frac{1}{M} \sum_{n, h, w} (X - \mu_c)^2 \quad (\text{biased normalizasyon varyansı})$$
  $$\hat{X} = \frac{X - \mu_c}{\sqrt{\sigma_c^2 + \epsilon_{\text{norm}}}}, \quad Y = \gamma_c \hat{X} + \beta_c$$
  *Hareketli İstatistikler (Running Stats, PyTorch Paritesi - Bessel Düzeltmesi):*
  $$\sigma_{\text{unbiased}}^2 = \sigma_c^2 \cdot \frac{M}{M - 1}$$
  $$\mu_{\text{run}} \leftarrow (1 - m_{\text{mom}}) \mu_{\text{run}} + m_{\text{mom}} \mu_c, \quad \sigma_{\text{run}}^2 \leftarrow (1 - m_{\text{mom}}) \sigma_{\text{run}}^2 + m_{\text{mom}} \sigma_{\text{unbiased}}^2$$
* **Değerlendirme (Eval) Modu:**
  $$Y = \gamma_c \frac{X - \mu_{\text{run}}}{\sqrt{\sigma_{\text{run}}^2 + \epsilon_{\text{norm}}}} + \beta_c$$

#### Geriye Yayılım Formülleri:
* **Parametre Gradyanları:**
  $$\frac{\partial L}{\partial \gamma_c} = \sum_{n, h, w} \delta(n, c, h, w) \cdot \hat{X}(n, c, h, w), \quad \frac{\partial L}{\partial \beta_c} = \sum_{n, h, w} \delta(n, c, h, w)$$
  *(Eval modunda parametre gradyanları aynı formülle hesaplanır; ancak $\hat{X}$ eğitim batch ortalamasıyla değil, $\mu_{\text{run}}$ ve $\sigma_{\text{run}}^2$ istatistikleriyle normalize edilir).*
* **Giriş Gradyanı (Eğitim Modu):**  
  $\hat{\delta} = \delta \odot \gamma$ tanımlandığında doğru analitik kapalı form:
  $$\frac{\partial L}{\partial X} = \frac{1}{M \sqrt{\sigma_c^2 + \epsilon_{\text{norm}}}} \left[ M \hat{\delta} - \sum_{i=1}^M \hat{\delta}_i - \hat{X} \sum_{i=1}^M (\hat{\delta}_i \cdot \hat{X}_i) \right]$$
* **Giriş Gradyanı (Eval Modu):**
  $$\frac{\partial L}{\partial X} = \delta \odot \frac{\gamma_c}{\sqrt{\sigma_{\text{run}}^2 + \epsilon_{\text{norm}}}}$$

---

### 8.2. LayerNorm
Normalizasyon batch ekseninde değil; her bir bağımsız örneğin kendi özellik boyutu ($D$) boyunca gerçekleşir. $\gamma \in \mathbb{R}^D$ normalize edilen eksen boyunca her eleman için değiştiğinden toplamın dışına çıkarılamaz.

#### İleri Besleme:
$$\mu = \frac{1}{D} \sum_{i=1}^D x_i, \quad \sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2$$
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon_{\text{norm}}}}, \quad y = \gamma \odot \hat{x} + \beta$$

#### Geriye Yayılım:
$\hat{\delta} = \delta \odot \gamma$ olmak üzere:
$$\frac{\partial L}{\partial x} = \frac{1}{D \sqrt{\sigma^2 + \epsilon_{\text{norm}}}} \left[ D \hat{\delta} - \sum_{i=1}^D \hat{\delta}_i - \hat{x} \sum_{i=1}^D (\hat{\delta}_i \cdot \hat{x}_i) \right]$$
$$\frac{\partial L}{\partial \gamma} = \sum_{\text{normalize edilmeyen eksenler (batch, seq)}} \delta \odot \hat{x} \quad \in \mathbb{R}^D$$
$$\frac{\partial L}{\partial \beta} = \sum_{\text{normalize edilmeyen eksenler}} \delta \quad \in \mathbb{R}^D$$

---

## 9. DÜZENLİLEŞTİRME: TERS ÖLÇEKLEMELİ DROPOUT (INVERTED DROPOUT)

* **İleri Besleme:** $p \in [0, 1)$ için $m_{\text{mask}} \sim \text{Bernoulli}(1 - p)$.
  $$Y = \begin{cases} \frac{X \odot m_{\text{mask}}}{1 - p} & \text{training} = \text{True} \\ X & \text{training} = \text{False} \end{cases}$$
* **Geriye Yayılım:** İleri geçişte üretilen **aynı maske** kullanılır:
  $$\nabla_X L = \begin{cases} \frac{\delta \odot m_{\text{mask}}}{1 - p} & \text{training} = \text{True} \\ \delta & \text{training} = \text{False} \end{cases}$$

---

## 10. ÇOK BAŞLIKLI ÖZ-DİKKAT MEKANİZMASI (MULTI-HEAD SELF-ATTENTION - MHSA)

Giriş tensörü $X \in \mathbb{R}^{B \times S_{\text{seq}} \times D}$. Başlık sayısı $H$, başlık başına boyut $d_k = D / H$.

### 10.1. İleri Besleme
1. Projeksiyonlar:
   $$Q = X W_Q + b_Q, \quad K = X W_K + b_K, \quad V = X W_V + b_V$$
   burada $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ ve $b_Q, b_K, b_V \in \mathbb{R}^D$.
2. Başlık Şekillendirme: $(B, S_{\text{seq}}, D) \to (B, S_{\text{seq}}, H, d_k) \xrightarrow{\text{transpose}} (B, H, S_{\text{seq}}, d_k)$.
3. Dikkat Skorları ve Ağırlıkları:
   $$\text{Scores} = \frac{Q @ K^T}{\sqrt{d_k}} \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times S_{\text{seq}}}$$
   $$A = \text{Softmax}(\text{Scores}, \text{axis}=-1)$$
4. Değer Toplamı (Head Context):
   $$\text{Head} = A @ V \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times d_k}$$
5. Başlık Birleştirme ve Çıkış Projeksiyonu:
   $$\text{Concat} = \text{Head}.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, S_{\text{seq}}, D)$$
   $$\text{Output} = \text{Concat} \cdot W_O + b_O, \quad W_O \in \mathbb{R}^{D \times D}, \quad b_O \in \mathbb{R}^D$$

### 10.2. Tam Analitik Geriye Yayılım Zinciri ($\delta = \frac{\partial L}{\partial \text{Output}}$)
* **Çıkış Katmanı Gradyanları:**
  $$\frac{\partial L}{\partial W_O} = \text{Concat}_{\text{flat}}^T \cdot \delta_{\text{flat}}, \quad \frac{\partial L}{\partial b_O} = \sum_{b=1}^B \sum_{s=1}^{S_{\text{seq}}} \delta(b, s, :)$$
  $$\delta_{\text{Concat}} = \delta @ W_O^T \quad \in \mathbb{R}^{B \times S_{\text{seq}} \times D}$$
* **Başlıklara Ayırma:**
  $$\delta_{\text{Head}} = \delta_{\text{Concat}}.\text{reshape}(B, S_{\text{seq}}, H, d_k).\text{transpose}(0, 2, 1, 3) \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times d_k}$$
* **V, A ve Softmax Geri Yayılımı:**
  $$\nabla V = A^T @ \delta_{\text{Head}} \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times d_k}$$
  $$\delta_A = \delta_{\text{Head}} @ V^T \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times S_{\text{seq}}}$$
  $$\delta_{\text{Scores}} = A \odot \left( \delta_A - \sum_{k} (\delta_A \odot A) \right)$$
  $$\nabla Q = \frac{1}{\sqrt{d_k}} (\delta_{\text{Scores}} @ K) \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times d_k}$$
  $$\nabla K = \frac{1}{\sqrt{d_k}} (\delta_{\text{Scores}}^T @ Q) \quad \in \mathbb{R}^{B \times H \times S_{\text{seq}} \times d_k}$$
* **Başlık Birleştirme (Head Merge):**
  Projeksiyon katmanlarına beslenmeden önce $\nabla Q, \nabla K, \nabla V$ $(B, S_{\text{seq}}, D)$ şekline geri dönüştürülür:
  $$\nabla Q_{\text{proj}} = \nabla Q.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, S_{\text{seq}}, D)$$
  $$\nabla K_{\text{proj}} = \nabla K.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, S_{\text{seq}}, D)$$
  $$\nabla V_{\text{proj}} = \nabla V.\text{transpose}(0, 2, 1, 3).\text{reshape}(B, S_{\text{seq}}, D)$$
* **Bias Gradyanları:**
  $$\frac{\partial L}{\partial b_Q} = \sum_{b, s} \nabla Q_{\text{proj}}, \quad \frac{\partial L}{\partial b_K} = \sum_{b, s} \nabla K_{\text{proj}}, \quad \frac{\partial L}{\partial b_V} = \sum_{b, s} \nabla V_{\text{proj}} \quad \in \mathbb{R}^D$$
* **Ağırlık Gradyanları:**
  $$\frac{\partial L}{\partial W_Q} = X_{\text{flat}}^T \cdot (\nabla Q_{\text{proj}})_{\text{flat}}, \quad \frac{\partial L}{\partial W_K} = X_{\text{flat}}^T \cdot (\nabla K_{\text{proj}})_{\text{flat}}, \quad \frac{\partial L}{\partial W_V} = X_{\text{flat}}^T \cdot (\nabla V_{\text{proj}})_{\text{flat}}$$
* **Giriş Tensörü $X$ Fan-Out Gradyan Toplamı:**
  $X$ üç ayrı projeksiyon yoluna ($Q, K, V$) çatallandığından autograd VJP toplam kuralı işletilir:
  $$\frac{\partial L}{\partial X} = \nabla Q_{\text{proj}} @ W_Q^T + \nabla K_{\text{proj}} @ W_K^T + \nabla V_{\text{proj}} @ W_V^T \quad \in \mathbb{R}^{B \times S_{\text{seq}} \times D}$$

---

## 11. KAYIP FONKSİYONLARI VE SAYISAL KARARLILIK (LOSS CALCULUS)

### 11.1. Ortalama Karesel Hata (MSELoss)
Toplam $M = \text{size}(Y)$ eleman için (`reduction="mean"`):
$$\mathcal{L} = \frac{1}{M} \sum (Y - \hat{Y})^2 \implies \frac{\partial \mathcal{L}}{\partial Y} = \frac{2}{M} (Y - \hat{Y})$$

---

### 11.2. Çapraz Entropi (CrossEntropyLoss)

#### Sayısal Taşma Sınırı (IEEE 754 FP32 Exp Sınırı):
Tek duyarlıklı kayan noktada (FP32) en büyük temsil edilebilir sonlu pozitif sayı $\approx 3.4028 \times 10^{38}$'dir ($2^{128} \times (1 - 2^{-24})$). Dolayısıyla üstel $e^z$ fonksiyonu $z > \ln(2^{128}-1) \approx \mathbf{88.72}$ olduğunda kesinlikle taşar (`+inf`). Bu eşik genel FP32 aritmetiği için değil, yalnızca $e^z$ üstel büyümesi için geçerli olan dinamik aralık sınırıdır.

#### Çözüm: Numerik Kararlı Log-Softmax:
$$\hat{z} = z - \max(z, \text{axis}=-1, \text{keepdims}=\text{True})$$
$$\log p_i = \hat{z}_i - \ln\left(\sum_j e^{\hat{z}_j}\right)$$

* **Sert Hedefler (Hard Targets / 1D Sınıf İndeksleri $y_n \in [0, C-1]$):**
  $$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \log p_{n, y_n}$$
* **Yumuşak Hedefler (Soft Targets / Label Smoothing $T \in \mathbb{R}^{N \times C}$, $\sum_c T_{n,c} = 1$):**
  $$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^N \sum_{c=1}^C T_{n, c} \log p_{n, c}$$

#### Analitik Birleşik Gradyan:
$$\frac{\partial \mathcal{L}}{\partial z} = \frac{p - T}{N}$$
> **Önemli Şart:** $\frac{p - T}{N}$ kapalı gradyan formülü **yalnızca** hedeflerin toplamı 1 olduğunda ($\sum_c T_c = 1$) analitik olarak doğrudur. 3D girişlerde ($B, S_{\text{seq}}, C$) normalizasyon $N = B \cdot S_{\text{seq}}$ üzerinden yürütülür.

---

## 12. OPTİMİZASYON ALGORİTMALARI (SGD MOMENTUM VE BAĞLI L2 ADAM)

### 12.1. SGD with Momentum & Weight Decay
$$g_t = \nabla_\theta \mathcal{L} + \lambda \theta_t \quad (\text{Bağlı L2 cezası})$$
$$v_t = \mu_{\text{sgd}} v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

### 12.2. Adam (Adaptive Moment Estimation)
$$g_t = \nabla_\theta \mathcal{L} + \lambda \theta_t$$
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon_{\text{adam}}} \hat{m}_t$$

> **Kritik Mimari Not (Adam vs AdamW):**  
> newton-vision içerisindeki Adam algoritması gradyana doğrudan $\lambda \theta_t$ ekler ($g_t \leftarrow g_t + \lambda \theta_t$). Bu nedenle `nv.Adam` **klasik bağlı (coupled) L2 ağırlık cezalı Adam'dır**. Ayrık (decoupled) ağırlık azaltma için `nv.AdamW` kullanılır: $\theta_t \leftarrow \theta_t (1 - \eta \lambda)$ uygulanır, ardından cezasız Adam adımı atılır (torch.optim.AdamW ile aynı, testle doğrulandı). Ayrıca weight decay parametre ayrımı gözetmeksizin `params` listesindeki tüm parametrelere (bias ve norm katmanları dahil) uygulanır.

---

## 13. INT8 KUANTİZASYON VE AFİN HARİTALAMA MATEMATİĞİ

### 13.1. Simetrik Kuantizasyon Formülü (newton-vision Standardı)
Ağırlıklar için sıfır noktası $Z = 0$, değer kümesi $[-127, 127]$ aralığına simetrik olarak sınırlandırılır ($-128$ asimetriden ve taşma dengesizliğinden kaçınmak için dahil edilmez):

$$\alpha = \max(|X|), \quad s_{\text{scale}} = \begin{cases} \frac{\alpha}{127.0} & \alpha > 0 \\ 1.0 & \alpha = 0 \end{cases}$$
$$q = \text{clip}\left( \left\lfloor \frac{X}{s_{\text{scale}}} \right\rceil, -127, 127 \right) \quad \in \text{int8}$$

* **Geri Kuantizasyon (Dequantization):**
  $$\hat{X} = q \cdot s_{\text{scale}} \quad \in \text{float32}$$

### 13.2. C++ SIMD vs NumPy Sayısal Paritesi
* NumPy `np.round`, Banker's Rounding (ties-to-even) uygular.
* C++ AVX2 motorunda tam bit paritesi sağlamak için `std::nearbyintf` veya `_mm256_round_ps(..., _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)` kullanılır. `std::roundf` sıfırdan uzağa yuvarladığı için tam $0.5$ eşiklerinde sapar.
* SIMD optimizasyonunda $X \cdot (1 / s_{\text{scale}})$ çarpımı yapılırsa FP32 son bit hassasiyeti (ULP) nedeniyle $0.5$ civarında farklı yöne yuvarlanabilir. Birebir eşitlik testi (`assert_array_equal`) için bölme işlemi korunur.

### 13.3. Motor Kapsamı (Quantization Scope)
newton-vision v1.1.6 sürümünde kuantizasyon kapsamı **ağırlık kuantizasyonu (Weight Quantization)** ve `QuantizedLinear` katmanında **INT32 akümülasyonlu integer matris çarpımıdır** ($Y = (X_{\text{int8}} @ W_{\text{int8}}) \cdot (s_x s_w) + b$). Uçtan uca INT8 aktivasyon evrişimi yol haritasındadır.

---

## 14. SAYISAL GRADYAN DOĞRULAMA VE GRADCHECK METODOLOJİSİ

### 14.1. İki Yönlü Merkezi Farklar (Central Differences)
$$g_{\text{num}} = \frac{f(\theta + \epsilon_{\text{step}}) - f(\theta - \epsilon_{\text{step}})}{2\epsilon_{\text{step}}}$$

### 14.2. FP32 Hata Dengesi ve Tolerans Sınırları (Merkezi Farklar)
Tek duyarlıkta (FP32) makine hassasiyeti $u \approx 6 \times 10^{-8}$'dir.
* **2. Dereceden Kesme Hatası (Taylor Truncation Error):** Merkezi fark formülü $O(\epsilon_{\text{step}}^2)$ kesme hatasına sahiptir: $\approx \frac{\epsilon_{\text{step}}^2}{6} f'''(\theta)$. *(Not: 1. derece ileri farklarda kesme hatası $O(\epsilon)$ olup optimum adım $u^{1/2} \approx 2.4 \times 10^{-4}$'tür).*
* **Yuvarlama Hatası (Rounding Cancellation):** $\approx \frac{u \cdot |f(\theta)|}{\epsilon_{\text{step}}}$.

Merkezi farklar için teorik toplam hatayı minimize eden adım boyutu:
$$\epsilon_{\text{step}}^* \approx u^{1/3} \approx (6 \times 10^{-8})^{1/3} \approx 4 \times 10^{-3}$$

Bu nedenle 2. dereceden merkezi farklarla yapılan FP32 sayısal gradcheck hesaplamalarında $\epsilon_{\text{step}} \approx 10^{-3} \dots 10^{-4}$ seçilir ve ulaşılabilecek teorik bağıl hata tabanı $10^{-3} \dots 10^{-4}$ mertebesindedir. $< 10^{-5}$ bağıl toleransı ise yalnızca çift duyarlıkta (FP64) aranabilir.

### 14.3. Test Kaybı Seçimi (Loss Selection Kuralı)
BatchNorm ve LayerNorm katmanlarında $\sum \hat{X} = 0$'dır. Dolayısıyla $L = \sum Y$ kaybı kullanılırsa:
* Girdi gradyanı $\frac{\partial L}{\partial X} = 0$ (özdeş olarak sıfır).
* Ölçek gradyanı $\frac{\partial L}{\partial \gamma} = \sum \hat{X} = 0$ (özdeş olarak sıfır).
* *Ancak* kayma gradyanı $\frac{\partial L}{\partial \beta} = \sum 1 = M$ sabit pozitif bir değerdir.

Girdi ve $\gamma$ gradyanlarının özdeş olarak sıfır çıkması türev implementasyon hatalarını gizler. Bu nedenle gradcheck testlerinde her zaman sabit rastgele bir projeksiyon vektörü $R$ ile skaler kayıp oluşturulmalıdır:
$$L = \sum (R \odot Y)$$

### 14.4. Gradcheck Metodoloji Kuralları
1. **Kıvrım Noktalarından Kaçınma (Kink Avoidance):** Türevi süreksiz olan noktalardan (örneğin ReLU $x = 0$, MaxPool eşit en büyük elemanlar) $\pm \epsilon_{\text{step}}$ aralığında uzak durulmalıdır.
2. **Parametre Başına Bağıl Hata:**
   $$\text{RelErr} = \frac{|g_{\text{num}} - g_{\text{ana}}|}{\max(|g_{\text{num}}|, |g_{\text{ana}}|) + 10^{-7}}$$
3. **Dropout Maskesinin Dondurulması:** Sayısal türev adımlarında ($\theta + \epsilon$ ve $\theta - \epsilon$) dropout maskesi sabit tutulmalıdır.
4. **Bileşik Çizge Testi:** Fan-out ve artık bağlantı hatalarını yakalamak için tekil katmanlar yerine çok katmanlı bloklar test edilmelidir.

---

*Bu döküman, newton-vision motorunun matematiksel şartnamesidir; `tests/test_math_spec.py` içerisindeki birim testleri ve `benchmarks/verify_parity_matrix.py` bünyesindeki 85 parametrik test bileşeni ile belirtilen test koşulları ve IEEE 754 sınırları dahilinde sayısal olarak doğrulanmaktadır.*
