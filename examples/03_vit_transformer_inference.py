"""
Örnek 03: Vision Transformer (ViT) Modeli ile Çıkarım Rehberi.
=============================================================================
Multi-Head Self-Attention (MHSA), LayerNorm ve GELU aktivasyonları içeren 
Vision Transformer (vit_tiny) modelinin kullanımını gösterir.
"""

import numpy as np
import newton_vision as nv


def main():
    print("=================================================================")
    print("🌌 newton-vision Örnek 03: Vision Transformer (ViT) Çıkarım Rehberi")
    print("=================================================================\n")

    # 1. ViT-Tiny Modeli (16x16 Yamalar, 1000 Sınıf)
    vit = nv.models.vit_tiny(num_classes=1000, img_size=224, patch_size=16)

    # 2. Sentetik Resim Girdisi (1, 3, 224, 224)
    img_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    img_tensor = nv.Tensor(img_data, requires_grad=False)

    print("▶ Vision Transformer (vit_tiny) Çıkarımı Çalıştırılıyor...")
    logits = vit(img_tensor)

    print(f"  ✓ ViT Logits Çıktı Boyutu     : {logits.data.shape}")
    print(f"  ✓ Tahmin Edilen Sınıf Indeksi: {np.argmax(logits.data)}")
    print("\n✅ ViT-Tiny Çıkarım İşlemi Başarıyla Doğrulandı!")


if __name__ == "__main__":
    main()
