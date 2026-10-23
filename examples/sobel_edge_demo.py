"""Canlı Görsel İşleme Demosu: Sobel Kenar Tespiti & Derin Özellik Çıkarımı.

Gerçek bir PNG resmi üretir, newton-vision C++ Conv2d Sobel filtrelerinden geçirir,
kenar haritasını (sobel_edges.png) diske kaydeder ve ResNet18/MobileNetV2 özellik haritasını çıkarır.
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageDraw
import newton_vision as nv

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


def create_sample_image(filename: str = "sample_input.png", size: int = 128) -> str:
    """128x128 boyutunda geometrik nesneler (daire, kare, çapraz çizgi) içeren resim üretir."""
    img = Image.new("L", (size, size), color=20)
    draw = ImageDraw.Draw(img)

    # Kare
    draw.rectangle([20, 20, 50, 50], fill=220, outline=255)
    # Daire
    draw.ellipse([70, 20, 110, 60], fill=180, outline=255)
    # Çapraz çizgi
    draw.line([20, 100, 100, 100], fill=255, width=4)

    filepath = os.path.join(DEMO_DIR, filename)
    img.save(filepath)
    print(f"  [1] Örnek giriş resmi oluşturuldu: {filepath}")
    return filepath


def run_sobel_edge_detection(input_path: str):
    """newton-vision C++ Conv2d çekirdeği ile Sobel Kenar Tespiti gerçekleştirir."""
    print("  [2] Resmi newton-vision Tensörüne aktarma (1, 1, 128, 128)...")
    img = Image.open(input_path).convert("L")
    img_arr = np.array(img, dtype=np.float32) / 255.0
    x = img_arr.reshape(1, 1, 128, 128)

    # Sobel X ve Sobel Y Filtreleri
    sobel_x_w = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
    sobel_y_w = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)

    conv_x = nv.Conv2d(1, 1, 3, padding=1, bias=False)
    conv_y = nv.Conv2d(1, 1, 3, padding=1, bias=False)

    conv_x.w.data = sobel_x_w
    conv_y.w.data = sobel_y_w

    print("  [3] newton-vision C++ Conv2d Sobel-X ve Sobel-Y filtrelerini çalıştırma...")
    gx = conv_x(x).data[0, 0]
    gy = conv_y(x).data[0, 0]

    # Magnitude: sqrt(gx^2 + gy^2)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude_norm = np.clip((magnitude / np.max(magnitude)) * 255.0, 0, 255).astype(np.uint8)

    output_path = os.path.join(DEMO_DIR, "sobel_edges.png")
    out_img = Image.fromarray(magnitude_norm)
    out_img.save(output_path)
    print(f"  [4] 🚀 C++ Conv2d filtrelenmiş kenar resmi diske kaydedildi: {output_path}")


def run_model_zoo_inference(input_path: str):
    """ResNet-18 ve MobileNetV2 modellerinde derin özellik çıkarımı."""
    print("\n  [5] ResNet-18 ve MobileNetV2 Derin Özellik Çıkarımı...")
    img = Image.open(input_path).convert("RGB").resize((224, 224))
    img_arr = (np.array(img, dtype=np.float32) / 255.0).transpose(2, 0, 1).reshape(1, 3, 224, 224)

    resnet = nv.models.resnet18(num_classes=1000)
    mobilenet = nv.models.mobilenet_v2(num_classes=1000)

    resnet_logits = resnet(img_arr)
    mobilenet_logits = mobilenet(img_arr)

    print(f"  ✓ ResNet-18 Çıktı Logit Tensör Boyutu    : {resnet_logits.shape}")
    print(f"  ✓ MobileNetV2 Çıktı Logit Tensör Boyutu : {mobilenet_logits.shape}")
    print(f"  ✓ ResNet-18 Maksimum Aktivasyon Değeri : {np.max(resnet_logits.data):.4f}")
    print(f"  ✓ MobileNetV2 Maksimum Aktivasyon Değeri: {np.max(mobilenet_logits.data):.4f}")


def main():
    print("==================================================")
    print("newton-vision: GERÇEK GÖRSEL İŞLEME & MODEL DEMOSU")
    print("==================================================")
    sample_path = create_sample_image()
    run_sobel_edge_detection(sample_path)
    run_model_zoo_inference(sample_path)
    print("\n✅ GERÇEK GÖRSEL DEMOSU BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
