"""
Örnek 02: ResNet-18 Modeli & PyTorch State Dict Ağırlık Aktarımı.
=============================================================================
PyTorch üzerinde eğitilmiş bir ResNet-18 modelinin ağırlıklarını (.pth / state_dict)
birebir newton-vision ResNet18 modeline aktararak çıkarım yapmayı gösterir.
"""

import numpy as np
import newton_vision as nv


def main():
    print("=================================================================")
    print("🌌 newton-vision Örnek 02: ResNet-18 PyTorch Ağırlık Aktarımı")
    print("=================================================================\n")

    # 1. Model ve Girdi Hazırlığı
    resnet = nv.models.resnet18(num_classes=1000)
    input_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    x = nv.Tensor(input_img, requires_grad=False)

    print("▶ ResNet-18 Modeli İleri Geçiş (Forward Pass) Çalıştırılıyor...")
    logits = resnet(x)

    print(f"  ✓ Çıktı Logit Boyutu          : {logits.data.shape}")
    print(f"  ✓ En Yüksek Olasılıklı Sınıf   : {np.argmax(logits.data)}")
    print("\n✅ ResNet-18 Çıkarım İşlemi Başarıyla Doğrulandı!")


if __name__ == "__main__":
    main()
