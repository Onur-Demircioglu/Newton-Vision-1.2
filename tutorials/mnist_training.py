"""Gerçek MNIST Veri Seti Eğitimi Rehberi (tutorials/mnist_training.py).

newton-vision ile LeNet sinir ağını MNIST el yazısı rakam veri kümesi üzerinde eğitir.
"""

from __future__ import annotations
import os
import urllib.request
import gzip
import numpy as np
import newton_vision as nv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_mnist():
    """MNIST veri setini NCBI/Yann LeCun yansısından indirir."""
    os.makedirs(DATA_DIR, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
    }
    for key, fname in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [+] Indiriliyor: {fname}...")
            try:
                urllib.request.urlretrieve(base_url + fname, fpath)
            except Exception as e:
                print(f"  [!] Indirme uyarisi: {e}")
                return None, None
    return files


def load_mnist_data():
    """Görüntüleri numpy dizisi olarak yükler."""
    files = download_mnist()
    if files is None or not os.path.exists(os.path.join(DATA_DIR, files["x_train"])):
        print("  [i] Sentetik MNIST benchmark verisi ile devam ediliyor...")
        X = np.random.randn(200, 1, 28, 28).astype(np.float32)
        Y = np.random.randint(0, 10, size=200).astype(np.int64)
        return X, Y

    with gzip.open(os.path.join(DATA_DIR, files["x_train"]), "rb") as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    with gzip.open(os.path.join(DATA_DIR, files["y_train"]), "rb") as f:
        Y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    return X, Y


def main():
    print("==================================================")
    print("newton-vision: GERÇEK MNIST VERİ KÜMESİ EĞİTİMİ")
    print("==================================================")

    X, Y = load_mnist_data()
    print(f"Veri Kümesi Yüklendi: {X.shape[0]} Örnek, Boyut: {X.shape[1:]}")

    class LeNetMNIST:
        def __init__(self):
            self.conv1 = nv.Conv2d(1, 6, kernel_size=5, padding=2)
            self.relu1 = nv.ReLU()
            self.pool1 = nv.MaxPool2d(2)
            self.conv2 = nv.Conv2d(6, 16, kernel_size=5)
            self.relu2 = nv.ReLU()
            self.pool2 = nv.MaxPool2d(2)
            self.fc1 = nv.Linear(16 * 5 * 5, 120)
            self.relu3 = nv.ReLU()
            self.fc2 = nv.Linear(120, 10)

        def forward(self, x):
            out = self.pool1(self.relu1(self.conv1(x)))
            out = self.pool2(self.relu2(self.conv2(out)))
            # Tensor.reshape grafı korur; out.data.reshape ham dizi verip konvolüsyonları eğitimden koparırdı
            flat = out.reshape(out.shape[0], -1)
            out = self.relu3(self.fc1(flat))
            return self.fc2(out)

    model = LeNetMNIST()
    criterion = nv.gravity.CrossEntropyLoss()
    optimizer = nv.gravity.Adam(model.conv1.parameters() + model.conv2.parameters() + model.fc1.parameters() + model.fc2.parameters(), lr=0.01)

    batch_size = 32
    epochs = 3

    print("\n🚀 Eğitime Başlanıyor (Batch Size=32)...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total_samples = min(X.shape[0], 500)

        for i in range(0, total_samples, batch_size):
            bx = nv.Tensor(X[i:i+batch_size], requires_grad=True)
            by = Y[i:i+batch_size]

            optimizer.zero_grad()
            logits = model.forward(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item() * len(by)
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == by)

        acc = (correct / total_samples) * 100.0
        avg_loss = total_loss / total_samples
        print(f"  Epoch [{epoch}/{epochs}] | Ortalama Loss: {avg_loss:.4f} | Doğruluk: %{acc:.1f}")

    print("\n✅ MNIST Eğitimi Başarıyla Tamamlandı!")


if __name__ == "__main__":
    main()
