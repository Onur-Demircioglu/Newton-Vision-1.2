"""Uçtan Uca Model Eğitimi, Metrikler & Karışıklık Matrisi Betiği (tutorials/train_and_eval_lenet.py).

newton-vision kütüphanesi ile LeNet evrişimsel sinir ağını MNIST üzerinde eğitir.
Eğitim sonrası Confusion Matrix, Precision, Recall, F1-Score, Class Accuracy, Top-1 ve Loss/Acc eğrilerini hesaplar.
"""

from __future__ import annotations
import os
import urllib.request
import gzip
import numpy as np
import newton_vision as nv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_mnist():
    os.makedirs(DATA_DIR, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "x_train": "train-images-idx3-ubyte.gz",
        "y_train": "train-labels-idx1-ubyte.gz",
        "x_test": "t10k-images-idx3-ubyte.gz",
        "y_test": "t10k-labels-idx1-ubyte.gz",
    }
    for key, fname in files.items():
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            try:
                urllib.request.urlretrieve(base_url + fname, fpath)
            except Exception as e:
                return None
    return files


def load_mnist():
    files = download_mnist()
    if files is None or not os.path.exists(os.path.join(DATA_DIR, files["x_train"])):
        x_tr = np.random.randn(500, 1, 28, 28).astype(np.float32)
        y_tr = np.random.randint(0, 10, size=500).astype(np.int64)
        x_te = np.random.randn(100, 1, 28, 28).astype(np.float32)
        y_te = np.random.randint(0, 10, size=100).astype(np.int64)
        return x_tr, y_tr, x_te, y_te

    def _read_images(fname):
        with gzip.open(os.path.join(DATA_DIR, fname), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def _read_labels(fname):
        with gzip.open(os.path.join(DATA_DIR, fname), "rb") as f:
            return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)

    x_tr = _read_images(files["x_train"])
    y_tr = _read_labels(files["y_train"])
    x_te = _read_images(files["x_test"])
    y_te = _read_labels(files["y_test"])

    return x_tr, y_tr, x_te, y_te


class LeNetClassifier(nv.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nv.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nv.ReLU()
        self.pool1 = nv.MaxPool2d(2)

        self.conv2 = nv.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nv.ReLU()
        self.pool2 = nv.MaxPool2d(2)

        self.fc1 = nv.Linear(16 * 5 * 5, 120)
        self.relu3 = nv.ReLU()
        self.fc2 = nv.Linear(120, 84)
        self.relu4 = nv.ReLU()
        self.fc3 = nv.Linear(84, 10)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))

        # Tensor.reshape grafı korur; ham .data ile yeni Tensor oluşturmak konvolüsyonların eğitilmesini engellerdi
        flat = out.reshape(out.shape[0], -1)

        out = self.relu3(self.fc1(flat))
        out = self.relu4(self.fc2(out))
        return self.fc3(out)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10):
    """Confusion Matrix, Precision, Recall, F1 ve Class Accuracy hesaplar."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precision = np.zeros(num_classes, dtype=np.float32)
    recall = np.zeros(num_classes, dtype=np.float32)
    f1 = np.zeros(num_classes, dtype=np.float32)
    acc_per_class = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        total_class = np.sum(cm[c, :])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision[c] = prec
        recall[c] = rec
        f1[c] = f1_val
        acc_per_class[c] = tp / total_class if total_class > 0 else 0.0

    return cm, precision, recall, f1, acc_per_class


def main():
    print("================================================================")
    print(" newton-vision: MNIST MODEL EGITIMI & DETAYLI METRIK RAPORU")
    print("================================================================")

    x_train, y_train, x_test, y_test = load_mnist()

    model = LeNetClassifier()
    criterion = nv.gravity.CrossEntropyLoss()
    optimizer = nv.gravity.Adam(model.parameters(), lr=0.001)  # 0.008, konvolüsyonlar da eğitilince kararsızdı

    batch_size = 64
    epochs = 5
    train_samples = min(x_train.shape[0], 2500)
    test_samples = min(x_test.shape[0], 1000)

    train_dataset = nv.TensorDataset(x_train[:train_samples], y_train[:train_samples])
    train_loader = nv.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = nv.TensorDataset(x_test[:test_samples], y_test[:test_samples])
    test_loader = nv.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    loss_history = []
    acc_history = []

    print("\n[INFO] DataLoader ile Egitime Baslaniyor (5 Epoch)...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        for bx, by in train_loader:
            bx.requires_grad = True

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item() * bx.shape[0]
            preds = np.argmax(logits.data, axis=1)
            correct += np.sum(preds == by.data)

        train_loss = total_loss / train_samples
        train_acc = (correct / train_samples) * 100.0

        loss_history.append(train_loss)
        acc_history.append(train_acc)

        print(f"  Epoch [{epoch}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.1f}")

    # TEST EVALUATION WITH no_grad()
    all_preds = []
    all_targets = y_test[:test_samples]

    with nv.no_grad():
        for bx, _ in test_loader:
            logits = model(bx)
            preds = np.argmax(logits.data, axis=1)
            all_preds.extend(preds)

    all_preds = np.array(all_preds[:test_samples], dtype=np.int64)

    # Compute Detailed Metrics
    cm, prec, rec, f1_scores, class_acc = compute_metrics(all_targets, all_preds)
    top1_acc = (np.sum(all_preds == all_targets) / test_samples) * 100.0

    print("\n" + "=" * 64)
    print(" EGITIM VE KAYIP EGRILERI (PROGRESS CURVES)")
    print("=" * 64)
    print(" Epoch  |  Train Loss  |  Train Accuracy")
    print(" -------+--------------+------------------")
    for ep, (l, a) in enumerate(zip(loss_history, acc_history), 1):
        bar = "#" * int(a / 5)
        print(f"   {ep:2d}   |    {l:6.4f}    |  %{a:5.1f}  {bar}")

    print("\n" + "=" * 64)
    print(f" GENEL TEST PERFORMANSI (TOP-1 ACCURACY): %{top1_acc:.2f}")
    print("=" * 64)

    print("\n" + "=" * 64)
    print(" SINIF BAZLI PERFORMANS TABLOSU (CLASS-WISE METRICS)")
    print("=" * 64)
    print(" Sinif (Rakam) | Precision | Recall   | F1-Score | Class Accuracy")
    print(" --------------+-----------+----------+----------+----------------")
    for c in range(10):
        print(f"   Sinif {c:1d}     |   {prec[c]:.4f}  |  {rec[c]:.4f}  |  {f1_scores[c]:.4f}  |  %{class_acc[c]*100:5.1f}")
    print(" --------------+-----------+----------+----------+----------------")
    print(f"   Ortalama    |   {np.mean(prec):.4f}  |  {np.mean(rec):.4f}  |  {np.mean(f1_scores):.4f}  |  %{np.mean(class_acc)*100:5.1f}")

    print("\n" + "=" * 64)
    print(" KARISIKLIK MATRISI (CONFUSION MATRIX 10x10)")
    print("=" * 64)
    print("     " + "".join([f"  P{c}" for c in range(10)]))
    for t in range(10):
        row_str = " ".join([f"{cm[t, p]:4d}" for p in range(10)])
        print(f" T{t} [{row_str}]")

    # apps/image_predictor.py'nin kullandığı lenet_mnist_v1.nv'nin üzerine yazmamak için ayrı dosya
    ckpt_path = os.path.join(os.path.dirname(__file__), "lenet_mnist_trained.nv")
    nv.save_model(model, ckpt_path)


if __name__ == "__main__":
    main()
