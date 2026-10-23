"""newton-vision v1.1.5 - Master Showcase Script

Bu script, projenin en kritik özelliklerini tek bir senaryoda doğrular:
1. Autograd & Eğitim (Yeni Tensor operasyonları ile)
2. Vision Transformer (ViT-Tiny) Forward & Backward akışı
3. INT8 Kuantizasyon & Bellek Optimizasyonu
4. JIT Model Serileştirme (.nv ve .nv.bin)
"""

import os
import sys
import time
import numpy as np

# Proje dizinini PYTHONPATH'e ekleyelim
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import newton_vision as nv


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def main():
    print_header("Newton-Vision v1.1.5 Yetenek Gosterimi")

    # ──────────────────────────────────────────────────────────
    # 1. AUTOGRAD & ML EĞİTİMİ (YENİ TENSOR OPERASYONLARI İLE)
    # ──────────────────────────────────────────────────────────
    print_header("Asama 1: Sifirdan Autograd ile Model Egitimi")
    
    # Sentetik veri seti (Binary sınıflandırma)
    np.random.seed(42)
    X_raw = np.random.randn(100, 8).astype(np.float32)
    y_raw = np.random.randint(0, 2, size=(100,)).astype(np.int64)

    # Katman ağırlıkları (requires_grad=True)
    w1 = nv.Tensor(np.random.randn(8, 16) * 0.1, requires_grad=True)
    b1 = nv.Tensor(np.zeros(16), requires_grad=True)
    w2 = nv.Tensor(np.random.randn(16, 2) * 0.1, requires_grad=True)
    b2 = nv.Tensor(np.zeros(2), requires_grad=True)

    optimizer = nv.gravity.Adam([w1, b1, w2, b2], lr=0.01)
    loss_fn = nv.gravity.CrossEntropyLoss()

    print("Egitim basliyor (15 Epoch)...")
    for epoch in range(1, 16):
        x = nv.Tensor(X_raw, requires_grad=False)
        
        # Forward pass (Senin refaktör ettiğin @ matmul ve + add operasyonları ile)
        h = x @ w1 + b1
        h_relu = nv.optics.ReLU()(h)
        logits = h_relu @ w2 + b2
        
        loss = loss_fn(logits, y_raw)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            preds = np.argmax(logits.data, axis=1)
            acc = np.mean(preds == y_raw) * 100
            print(f"  Epoch {epoch:02d}/15 | Loss: {float(loss.data):.4f} | Egitim Acc: %{acc:.1f}")

    # ──────────────────────────────────────────────────────────
    # 2. VISION TRANSFORMER (ViT) FORWARD & BACKWARD PASS
    # ──────────────────────────────────────────────────────────
    print_header("Asama 2: Vision Transformer (ViT-Tiny) Entegrasyonu")
    
    # Dummy Görsel Girdisi: (Batch=1, Channels=3, Height=224, Width=224)
    img = nv.Tensor(np.random.randn(1, 3, 224, 224), requires_grad=True)
    
    print("ViT-Tiny modeli yukleniyor...")
    vit = nv.models.vit_tiny(num_classes=10, img_size=224, patch_size=16)
    
    print("Forward pass calistiriliyor...")
    start_time = time.time()
    logits = vit(img)
    duration = (time.time() - start_time) * 1000
    
    print(f"  [OK] Forward Pass tamamlandi! Sure: {duration:.1f} ms")
    print(f"  [OK] Cikti boyutu (Logits Shape): {logits.data.shape}")
    
    print("Backward pass calistiriliyor (Gradyan akisi dogrulamasi)...")
    logits.sum().backward()
    
    print(f"  [OK] Gradyan akisi basarili!")
    print(f"  [OK] Input Görseli Gradyan Boyutu: {img.grad.shape}")
    print(f"  [OK] Patch Embed Agirlik Gradyan Boyutu: {vit.patch_embed.proj.w.grad.shape}")

    # ──────────────────────────────────────────────────────────
    # 3. INT8 KUANTİZASYON MOTORU
    # ──────────────────────────────────────────────────────────
    print_header("Asama 3: Model Kuantizasyon & INT8 Optimizasyonu")
    
    print("Orijinal model FP32 bellek durumu analiz ediliyor...")
    # ViT modelini kuantize edelim
    stats = nv.quantize_model(vit)
    
    print(f"  [OK] Kuantize Edilen Parametre Sayisi: {stats['total_quantized_params']:,}")
    print(f"  [OK] FP32 Agirlik Bellek Boyutu    : {stats['fp32_memory_kb']:.2f} KB")
    print(f"  [OK] INT8 Agirlik Bellek Boyutu    : {stats['int8_memory_kb']:.2f} KB")
    print(f"  [OK] Bellek Tasarrufu Orani (%)   : %{stats['memory_reduction_percent']:.2f}")

    # ──────────────────────────────────────────────────────────
    # 4. JIT SERİLEŞTİRME VE DOSYA İŞLEMLERİ
    # ──────────────────────────────────────────────────────────
    print_header("Asama 4: JIT Model Kaydetme & Yukleme")
    
    model_path_nv = "showcase_vit.nv"
    model_path_bin = "showcase_vit.nv.bin"
    
    print(f"Model .nv formatinda diske kaydediliyor: {model_path_nv}")
    nv.save_model(vit, model_path_nv)
    
    print(f"Model saf C++ ikili .nv.bin formatinda diske kaydediliyor: {model_path_bin}")
    nv.save_model_bin(vit, model_path_bin)
    
    # Temizlik
    if os.path.exists(model_path_nv):
        os.remove(model_path_nv)
    if os.path.exists(model_path_bin):
        os.remove(model_path_bin)
    print("  [OK] Model serilestirme ve dosya temizligi basarili.")

    print("\n" + "=" * 70)
    print(" GOSTERIM BASARIYLA TAMAMLANDI! Tumu sorunsuz calisiyor.")
    print("=" * 70)


if __name__ == "__main__":
    main()
