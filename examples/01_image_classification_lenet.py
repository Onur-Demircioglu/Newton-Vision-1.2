"""
Örnek 01: Mini CNN (Conv -> ReLU -> MaxPool -> Flatten -> Linear) eğitim API'si.
=============================================================================
MNIST boyutunda (1x28x28) **sentetik, rastgele etiketli** veriyle eğitim döngüsünün
API'sini gösterir; düşen kayıp modelin veriyi ezberlediğini gösterir, genellemeyi değil.
Gerçek MNIST eğitimi için: tutorials/train_and_eval_lenet.py veya tutorials/01_train_mnist.py
"""

import numpy as np
import newton_vision as nv


def main():
    print("=================================================================")
    print("🌌 newton-vision Örnek 01: Mini CNN Eğitim API'si (sentetik veri)")
    print("=================================================================\n")

    # 1. Sentetik Veri Seti Oluşturma (100 Görsel, 1x28x28, rastgele etiket)
    np.random.seed(42)
    X_train = np.random.randn(100, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(100,)).astype(np.int64)

    # 2. Model Katmanlarının Tanımlanması
    conv1 = nv.optics.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
    relu1 = nv.optics.ReLU()
    pool1 = nv.optics.MaxPool2d(kernel_size=2, stride=2)
    flatten = nv.Flatten()
    fc1 = nv.optics.Linear(in_features=4 * 14 * 14, out_features=10)

    params = [conv1.w, conv1.bias, fc1.w, fc1.bias]
    optimizer = nv.gravity.Adam(params, lr=0.01)

    print("▶ Eğitim Başlatılıyor (20 Epoch)...")
    for epoch in range(1, 21):
        x_tensor = nv.Tensor(X_train, requires_grad=True)
        
        # Forward pass
        h1 = relu1(conv1(x_tensor))
        h2 = pool1(h1)
        h2_flat = flatten(h2)  # grafı korur: gradyan conv1'e kadar akar
        logits = fc1(h2_flat)

        # Cross Entropy Loss
        loss_fn = nv.gravity.CrossEntropyLoss()
        loss = loss_fn(logits, y_train)

        # Backward Pass & Gradient Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  • Epoch {epoch:02d}/20 | Kayıp (Loss): {float(loss.data):.4f}")

    print("\n✅ Eğitim Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
