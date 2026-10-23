"""
tutorials/01_train_mnist.py - Tekrarlanabilir (Reproducible) Gerçek MNIST Eğitimi.
=============================================================================
Bu öğretici, newton-vision ile gerçek MNIST verisi üzerinde küçük bir CNN eğitir ve
FP32 ile INT8 kuantize model arasındaki doğruluk farkını (Accuracy Drop) ölçer.

TEKRARLANABİLİRLİK (REPRODUCIBILITY) PARAMETRELERİ:
- Veri Seti           : MNIST (tutorials/data/*.gz; yoksa önce tutorials/train_and_eval_lenet.py indirir)
- Eğitim Örnek Sayısı : 6000 (eğitim kümesinin ilk 6000 görseli)
- Test Örnek Sayısı   : 1000 (test kümesinin ilk 1000 görseli)
- Epoch Sayısı        : 2
- Batch Size          : 64 (karıştırmalı mini-batch)
- Optimizatör         : Adam (lr=0.001)
- Random Seed         : 42
"""

import gzip
import os
import numpy as np
import newton_vision as nv

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NUM_TRAIN, NUM_TEST = 6000, 1000
EPOCHS, BATCH_SIZE, LR, SEED = 2, 64, 1e-3, 42
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _read_idx(fname: str, offset: int) -> np.ndarray:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MNIST dosyası bulunamadı: {path}. İndirmek için önce `python tutorials/train_and_eval_lenet.py` çalıştırın."
        )
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)


def load_mnist():
    x_tr = _read_idx("train-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)[:NUM_TRAIN]
    y_tr = _read_idx("train-labels-idx1-ubyte.gz", 8)[:NUM_TRAIN].astype(np.int64)
    x_te = _read_idx("t10k-images-idx3-ubyte.gz", 16).reshape(-1, 1, 28, 28)[:NUM_TEST]
    y_te = _read_idx("t10k-labels-idx1-ubyte.gz", 8)[:NUM_TEST].astype(np.int64)
    normalize = lambda x: ((x.astype(np.float32) / 255.0) - MNIST_MEAN) / MNIST_STD
    return normalize(x_tr), y_tr, normalize(x_te), y_te


def build_model() -> nv.Sequential:
    return nv.Sequential(
        nv.Conv2d(1, 8, kernel_size=3, padding=1),
        nv.ReLU(),
        nv.MaxPool2d(kernel_size=2, stride=2),
        nv.Flatten(),                    # grafı korur: gradyan konvolüsyona kadar akar
        nv.Linear(8 * 14 * 14, 10),
    )


def evaluate(model, x, y) -> float:
    model.eval()
    with nv.no_grad():
        preds = np.concatenate([model(nv.Tensor(x[i:i + 500])).data.argmax(axis=1) for i in range(0, len(x), 500)])
    return float(np.mean(preds == y) * 100.0)


def run_reproducible_experiment():
    print("=================================================================")
    print("🎓 newton-vision Öğretici 01: Tekrarlanabilir Gerçek MNIST Eğitimi")
    print("=================================================================")
    print(f"• Random Seed        : {SEED}")
    print(f"• Veri               : {NUM_TRAIN} eğitim / {NUM_TEST} test (gerçek MNIST)")
    print(f"• Optimizatör        : Adam (lr={LR})")
    print(f"• Epochs / Batch     : {EPOCHS} / {BATCH_SIZE}")
    print("=================================================================\n")

    np.random.seed(SEED)
    x_train, y_train, x_test, y_test = load_mnist()

    model = build_model()
    optimizer = nv.Adam(model.parameters(), lr=LR)
    loss_fn = nv.CrossEntropyLoss()
    loader = nv.DataLoader(nv.TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    conv_w_initial = model[0].w.data.copy()

    print("▶ FP32 Model Eğitimi Başlatılıyor...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for bx, by in loader:
            loss = loss_fn(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            seen += len(by)
        print(f"  • Epoch {epoch}/{EPOCHS} | Loss: {total_loss / seen:.4f} | Test Acc: %{evaluate(model, x_test, y_test):.2f}")

    conv_change = float(np.abs(model[0].w.data - conv_w_initial).max())
    if conv_change == 0.0:
        raise RuntimeError("Konvolüsyon ağırlıkları değişmedi: gradyan grafı kopuk!")
    fp32_acc = evaluate(model, x_test, y_test)

    # INT8 Kuantizasyon Deneyi: aynı ağırlıklarla ayrı bir model, Conv2d/Linear katmanları INT8'e çevrilir
    print("\n▶ INT8 Kuantizasyon Deneyi İşletiliyor (nv.quantize_model)...")
    int8_model = build_model()
    int8_model.load_state_dict(model.state_dict(), strict=True)
    stats = nv.quantize_model(int8_model, per_channel=True)
    int8_acc = evaluate(int8_model, x_test, y_test)

    print("\n=================================================================")
    print("📊 DENEY SONUÇLARI (EXPERIMENT #01):")
    print("=================================================================")
    print(f"  • Konvolüsyon ağırlık değişimi (max)  : {conv_change:.4f}")
    print(f"  • FP32 Test Doğruluğu (Accuracy)      : %{fp32_acc:.2f}")
    print(f"  • INT8 Test Doğruluğu (Accuracy)      : %{int8_acc:.2f}")
    print(f"  • Kuantizasyon Farkı (Accuracy Drop)  : {fp32_acc - int8_acc:+.2f} puan")
    print(f"  • Ağırlık Belleği                     : {stats['fp32_memory_kb']:.1f} KB -> {stats['int8_memory_kb']:.1f} KB")
    print("=================================================================\n")

    print("✅ Tekrarlanabilir MNIST Öğreticisi Başarıyla Tamamlandı!")


if __name__ == "__main__":
    run_reproducible_experiment()
