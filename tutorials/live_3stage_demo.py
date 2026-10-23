"""
tutorials/live_3stage_demo.py - 3 Aşamalı Canlı Uygulama ve Karşılaştırma Testi.
=============================================================================
Aşama 1: PyTorch vs newton-vision Canlı Karşılaştırma Testi.
Aşama 2: PyTorch ile Birlikte (Joint / Hybrid) Kullanım (PyTorch Weight -> nv.INT8 Quantized Engine).
Aşama 3: Sadece newton-vision ile Sıfırdan Model Eğitimi (Pure Training).
"""

import time
import numpy as np
import torch
import torch.nn as tnn
import newton_vision as nv

try:
    import torchvision.models as torch_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


def stage_1_direct_comparison():
    print("=================================================================")
    print("🔥 AŞAMA 1: PyTorch vs newton-vision Canlı Karşılaştırma Testi")
    print("=================================================================\n")

    input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # 1. Vision Transformer (ViT-Tiny) Testi
    nv_vit = nv.models.vit_tiny(num_classes=1000)
    x_nv = nv.Tensor(input_data, requires_grad=False)

    start = time.perf_counter()
    out_nv = nv_vit(x_nv)
    nv_time = (time.perf_counter() - start) * 1000.0

    print(f"  ✓ newton-vision ViT-Tiny Çıkarım Süresi: {nv_time:.2f} ms | Çıktı Boyutu: {out_nv.data.shape}")

    # 2. PyTorch ResNet-18
    if HAS_TORCHVISION:
        t_resnet = torch_models.resnet18(weights=None).eval()
        x_t = torch.from_numpy(input_data)

        start = time.perf_counter()
        with torch.no_grad():
            out_t = t_resnet(x_t)
        t_time = (time.perf_counter() - start) * 1000.0

        print(f"  ✓ PyTorch ResNet-18 Çıkarım Süresi     : {t_time:.2f} ms | Çıktı Boyutu: {out_t.shape}")
    else:
        # Local PyTorch ConvNet
        t_net = tnn.Sequential(
            tnn.Conv2d(3, 16, 3, padding=1),
            tnn.ReLU(),
            tnn.AdaptiveAvgPool2d((1, 1)),
            tnn.Flatten(),
            tnn.Linear(16, 1000)
        ).eval()
        x_t = torch.from_numpy(input_data)
        start = time.perf_counter()
        with torch.no_grad():
            out_t = t_net(x_t)
        t_time = (time.perf_counter() - start) * 1000.0
        print(f"  ✓ PyTorch ConvNet Çıkarım Süresi       : {t_time:.2f} ms | Çıktı Boyutu: {out_t.shape}")

    print("\n✅ Aşama 1 Başarıyla Tamamlandı!")


def stage_2_joint_hybrid_usage():
    print("\n=================================================================")
    print("🤝 AŞAMA 2: PyTorch & newton-vision Hibrit (Birlikte) Kullanım")
    print("=================================================================\n")

    # 1. PyTorch Üzerinde Basit Bir Model Tanımlama ve Ağırlık Kaydetme
    class SimplePyTorchNet(tnn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tnn.Linear(10, 5)

        def forward(self, x):
            return self.fc1(x)

    torch_net = SimplePyTorchNet()
    state_dict = torch_net.state_dict()

    print("  • PyTorch Tarafında Model Eğitildi / Ağırlıklar Hazırlandı.")

    # 2. newton-vision Tarafında Ağırlıkları Yükleme & INT8 Kuantizasyonu İşletme
    nv_linear = nv.optics.Linear(in_features=10, out_features=5)
    nv.load_state_dict(nv_linear, {"w": state_dict["fc1.weight"].numpy().T, "bias": state_dict["fc1.bias"].numpy()})

    print("  ✓ PyTorch Ağırlıkları newton-vision Katmanına Aktarıldı.")

    # INT8 Kuantizasyonu
    q_linear = nv.QuantizedLinear(in_features=10, out_features=5)
    q_linear.load_fp32_weights(state_dict["fc1.weight"].numpy().T, state_dict["fc1.bias"].numpy())

    dummy_input = np.random.randn(2, 10).astype(np.float32)
    out_q = q_linear(dummy_input)

    print(f"  ✓ PyTorch Ağırlıkları INT8 Kuantize Edilerek Çıkarım Yapıldı | Çıktı Boyutu: {out_q.data.shape}")
    print("\n✅ Aşama 2 (Hibrit Kullanım) Başarıyla Tamamlandı!")


def stage_3_pure_newton_vision_training():
    print("\n=================================================================")
    print("🌌 AŞAMA 3: Sadece newton-vision ile Sıfırdan Model Eğitimi")
    print("=================================================================\n")

    np.random.seed(42)
    # Sentetik Veri Seti (80 Örnek, 16 Özellik, 3 Sınıf)
    X = np.random.randn(80, 16).astype(np.float32)
    y = np.random.randint(0, 3, size=(80,)).astype(np.int64)

    # newton-vision Çöküntüsüz (Dense) Model Mimarisi
    fc1 = nv.optics.Linear(16, 32)
    relu1 = nv.optics.ReLU()
    fc2 = nv.optics.Linear(32, 3)

    params = [fc1.w, fc1.bias, fc2.w, fc2.bias]
    optimizer = nv.gravity.Adam(params, lr=0.02)
    loss_fn = nv.gravity.CrossEntropyLoss()

    print("▶ Sıfırdan newton-vision Autograd Eğitimi Başlatılıyor (30 Epoch)...")
    initial_loss = 0.0

    for epoch in range(1, 31):
        x_tensor = nv.Tensor(X, requires_grad=True)

        # Forward Pass
        h1 = relu1(fc1(x_tensor))
        logits = fc2(h1)

        # Loss Calculation
        loss = loss_fn(logits, y)

        if epoch == 1:
            initial_loss = float(loss.data)

        # Autograd Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  • Epoch {epoch:02d}/30 | Kayıp (Loss): {float(loss.data):.4f}")

    final_loss = float(loss.data)
    loss_drop = ((initial_loss - final_loss) / initial_loss) * 100.0

    # Accuracy calculation
    preds = np.argmax(logits.data, axis=1)
    acc = np.mean(preds == y) * 100.0

    print(f"\n  🏆 Başlangıç Kaybı : {initial_loss:.4f}")
    print(f"  🏆 Bitiş Kaybı     : {final_loss:.4f} (Kayıp Düşüşü: %{loss_drop:.2f})")
    print(f"  🏆 Model Eğitilmiş Doğruluğu (Accuracy): %{acc:.2f}")

    print("\n✅ Aşama 3 (Sadece newton-vision ile Saf Eğitim) Başarıyla Tamamlandı!")


if __name__ == "__main__":
    stage_1_direct_comparison()
    stage_2_joint_hybrid_usage()
    stage_3_pure_newton_vision_training()
