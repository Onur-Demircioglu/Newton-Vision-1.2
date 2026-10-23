"""
Örnek 05: Eğitilen Modeli ONNX Formatına Dışa Aktarma (Export).
=============================================================================
newton-vision modellerini standart .onnx formatına aktararak ONNX Runtime veya 
diğer çıkarım motorlarında çalıştırma kılavuzu.
"""

import os
import numpy as np
import newton_vision as nv


def main():
    print("=================================================================")
    print("🌌 newton-vision Örnek 05: ONNX Formatına Model Aktarımı")
    print("=================================================================\n")

    model = nv.models.resnet18(num_classes=1000)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    output_path = "resnet18_exported.onnx"

    print(f"▶ ResNet-18 Modeli ONNX Formatına Aktarılıyor -> '{output_path}'...")
    nv.jit.export_onnx(model, dummy_input, output_path)

    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024.0
        print(f"  ✓ ONNX Dosyası Başarıyla Oluşturuldu ({size_kb:.2f} KB)")
        os.remove(output_path)

    print("\n✅ ONNX Export İşlemi Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
