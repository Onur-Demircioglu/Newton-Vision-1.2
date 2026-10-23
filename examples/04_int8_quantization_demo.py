"""
Örnek 04: INT8 Kuantizasyon ile %75 Bellek Tasarrufu Demo.
=============================================================================
FP32 ağırlıklı derin bir modelin Per-Channel INT8 kuantizasyon motoru ile 
küçültülmesini ve bellek tasarrufunu gösterir.
"""

import newton_vision as nv


def main():
    print("=================================================================")
    print("🌌 newton-vision Örnek 04: INT8 Kuantizasyon ile Bellek Küçültme")
    print("=================================================================\n")

    # 1. ResNet-18 FP32 Modeli
    model = nv.models.resnet18(num_classes=1000)

    # 2. INT8 Kuantizasyonu İşlemi
    print("▶ Model Ağırlıkları Per-Channel INT8 Formatına Dönüştürülüyor...")
    stats = nv.quantize_model(model, per_channel=True)

    print(f"  ✓ Kuantize Edilen Parametreler: {stats['total_quantized_params']:,}")
    print(f"  ✓ FP32 Orijinal Bellek Boyutu : {stats['fp32_memory_kb']:.2f} KB")
    print(f"  ✓ INT8 Kuantize Bellek Boyutu : {stats['int8_memory_kb']:.2f} KB")
    print(f"  🏆 BELLEK TASARRUFU ORANI (%) : %{stats['memory_reduction_percent']:.2f}")

    print("\n✅ INT8 Model Kuantizasyonu Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
