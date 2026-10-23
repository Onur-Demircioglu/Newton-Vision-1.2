"""LeNet CNN Görsel Sınıflandırma Modeli Eğitimi (examples/train_mnist.py).

newton-vision kütüphanesinin Conv2d, ReLU, MaxPool2d katmanlarını, Fluxion Autograd motorunu
ve Gravity Adam optimizer'ını kullanarak uçtan uca görsel sınıflandırma modeli eğitir.
"""

from __future__ import annotations
import numpy as np
import newton_vision as nv


def generate_digit_dataset(num_samples: int = 100):
    """Sentetik 8x8 görsel rakam veri seti üretici.
    Sınıf 0: Dikey Çizgi (Vertical Line)
    Sınıf 1: Yatay Çizgi (Horizontal Line)
    """
    X = np.zeros((num_samples, 1, 8, 8), dtype=np.float32)
    Y = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        label = i % 2
        Y[i] = label
        img = np.random.randn(8, 8).astype(np.float32) * 0.1
        if label == 0:
            # Dikey çizgi (Sınıf 0)
            img[:, 3:5] += 2.0
        else:
            # Yatay çizgi (Sınıf 1)
            img[3:5, :] += 2.0
        X[i, 0] = img

    return X, Y


class LeNetMini:
    """Mini LeNet Evrişimsel Sinir Ağı."""
    def __init__(self):
        self.conv1 = nv.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nv.ReLU()
        self.pool1 = nv.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4

        self.conv2 = nv.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nv.ReLU()
        self.pool2 = nv.MaxPool2d(kernel_size=2, stride=2)  # 4x4 -> 2x2

        # 8 kanal x 2x2 = 32 özellik -> 2 sınıf
        scale = np.sqrt(2.0 / 32, dtype=np.float32)
        self.fc_w = nv.Tensor(np.random.randn(32, 2).astype(np.float32) * scale, requires_grad=True)
        self.fc_b = nv.Tensor(np.zeros(2, dtype=np.float32), requires_grad=True)

    def parameters(self) -> list[nv.Tensor]:
        return self.conv1.parameters() + self.conv2.parameters() + [self.fc_w, self.fc_b]

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))

        # Flatten (N, 8, 2, 2) -> (N, 32)
        N = out.data.shape[0]
        flat_data = out.data.reshape(N, 32)
        flat_tensor = nv.Tensor(flat_data, requires_grad=out.requires_grad, _children=(out,))

        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    out.grad = flat_tensor.grad.reshape(out.data.shape)
                else:
                    out.grad += flat_tensor.grad.reshape(out.data.shape)
            flat_tensor._backward = _backward

        # FC layer: logits = flat @ w + b
        logits_data = flat_tensor.data @ self.fc_w.data + self.fc_b.data
        logits = nv.Tensor(logits_data, requires_grad=True, _children=(flat_tensor, self.fc_w, self.fc_b))

        def _backward_fc():
            if flat_tensor.requires_grad:
                g_flat = logits.grad @ self.fc_w.data.T
                flat_tensor.grad = flat_tensor.grad + g_flat if flat_tensor.grad is not None else g_flat
            if self.fc_w.requires_grad:
                g_w = flat_tensor.data.T @ logits.grad
                self.fc_w.grad = self.fc_w.grad + g_w if self.fc_w.grad is not None else g_w
            if self.fc_b.requires_grad:
                g_b = np.sum(logits.grad, axis=0)
                self.fc_b.grad = self.fc_b.grad + g_b if self.fc_b.grad is not None else g_b

        logits._backward = _backward_fc
        return logits


def main():
    print("==================================================")
    print("newton-vision: LeNet Görsel Sınıflandırma Eğitimi")
    print("==================================================")
    print(f"Cihaz: {nv.principia.get_device().upper()} (CUDA Hazır: {nv.principia.is_cuda_available()})")

    X, Y = generate_digit_dataset(num_samples=100)
    print(f"Veri Seti: {X.shape[0]} örnek, 8x8 piksel görseller (Sınıf 0: Dikey, Sınıf 1: Yatay)\n")

    model = LeNetMini()
    criterion = nv.gravity.CrossEntropyLoss()
    optimizer = nv.gravity.Adam(model.parameters(), lr=0.03)

    epochs = 30
    batch_size = 20

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        # Batch döngüsü
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)

        for i in range(0, X.shape[0], batch_size):
            batch_idx = indices[i:i + batch_size]
            bx = nv.Tensor(X[batch_idx], requires_grad=True)
            by = Y[batch_idx]

            optimizer.zero_grad()
            logits = model.forward(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item() * len(batch_idx)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == by)

        avg_loss = total_loss / X.shape[0]
        acc = (correct / X.shape[0]) * 100.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:2d}/{epochs}] | Ort. Kayıp (Loss): {avg_loss:.4f} | Doğruluk (Accuracy): %{acc:.1f}")

    print("\n✅ LeNet Modeli Başarıyla Eğitildi!")
    print(f"🏆 Son Sınıflandırma Doğruluğu: %{acc:.1f}")


if __name__ == "__main__":
    main()
