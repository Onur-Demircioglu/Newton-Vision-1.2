"""CLAUDE.md Madde 10: Doğrusal Regresyon Matematiksel Yakınsama Doğrulaması.

Gerçek denklem: y = 3.0 * x + 2.0
Hedef: newton-vision bu ağırlıkları sıfırdan öğrenip w ≈ 3.0, b ≈ 2.0 bulacak mı?
"""

import unittest
import numpy as np
import newton_vision as nv

def test_linear_regression_convergence():
    print("\n========================================================")
    print(" DENEY 1: DETERMINISTIK DOGRUSAL REGRESYON (y = 3x + 2)")
    print("========================================================")

    np.random.seed(42)
    # 200 adet örnek
    X = np.linspace(-2, 2, 200).reshape(-1, 1).astype(np.float32)
    # Gürültüsüz kesin zemin gerçeği (Ground Truth): w=3.0, b=2.0
    Y = 3.0 * X + 2.0

    dataset = nv.TensorDataset(X, Y)
    loader = nv.DataLoader(dataset, batch_size=32, shuffle=True)

    model = nv.Linear(1, 1, bias=True)
    # Başlangıçta ağırlıklar rastgeledir
    print(f"  [Baslangic] w: {model.w.data[0,0]:.4f} | b: {model.bias.data[0]:.4f}")

    criterion = nv.gravity.MSELoss()
    optimizer = nv.gravity.SGD(model.parameters(), lr=0.05, momentum=0.9)

    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(1, 101):
        epoch_loss = 0.0
        for bx, by in loader:
            bx.requires_grad = True
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data) * len(bx.data)

        epoch_loss /= len(X)
        if epoch == 1:
            initial_loss = epoch_loss
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/100 | Loss: {epoch_loss:.6f} | w: {model.w.data[0,0]:.4f} | b: {model.bias.data[0]:.4f}")

    final_loss = epoch_loss
    learned_w = float(model.w.data[0, 0])
    learned_b = float(model.bias.data[0])

    print("\n  [SONUCLAR]")
    print(f"  Hedef W     : 3.0000 | Ogrenilen W: {learned_w:.4f} | Fark: {abs(3.0 - learned_w):.5f}")
    print(f"  Hedef B     : 2.0000 | Ogrenilen B: {learned_b:.4f} | Fark: {abs(2.0 - learned_b):.5f}")
    print(f"  Loss Dususu : {initial_loss:.4f} -> {final_loss:.8f}")

    assert abs(3.0 - learned_w) < 0.05, f"W yakinsamadi: {learned_w}"
    assert abs(2.0 - learned_b) < 0.05, f"B yakinsamadi: {learned_b}"
    print("  [BASARILI] Model parametreleri tam olarak ogrendi!\n")


def test_xor_nonlinear_convergence():
    print("========================================================")
    print(" DENEY 2: DOGRUSAL OLMAYAN XOR PROBLEMI (MLP)")
    print("========================================================")
    # XOR doğrusal olarak ayrılamaz, gizli katman ve aktivasyonun çalıştığını kanıtlar
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

    model = nv.Sequential(
        nv.Linear(2, 8),
        nv.ReLU(),
        nv.Linear(8, 1),
        nv.Sigmoid()
    )

    criterion = nv.gravity.MSELoss()
    optimizer = nv.gravity.Adam(model.parameters(), lr=0.05)

    print("  XOR Egitimi basliyor...")
    for epoch in range(1, 301):
        bx = nv.Tensor(X, requires_grad=True)
        by = nv.Tensor(Y, requires_grad=False)

        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}/300 | Loss: {float(loss.data):.6f}")

    with nv.no_grad():
        final_preds = model(nv.Tensor(X)).data

    print("\n  [XOR TAHMINLERI]")
    for i in range(4):
        print(f"  Girdi: {X[i]} | Hedef: {Y[i][0]} | Model Ciktisi: {final_preds[i][0]:.4f}")

    # Eşik değer 0.5 ile doğrula
    binary_preds = (final_preds > 0.5).astype(np.float32)
    np.testing.assert_allclose(binary_preds, Y)
    print("  [BASARILI] XOR tamamen cozuldu (%100 Dogruluk)!\n")


def load_tests(loader, tests, pattern):
    """`python -m unittest discover` modül düzeyindeki test_* fonksiyonlarını da çalıştırsın diye."""
    funcs = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in funcs)


if __name__ == "__main__":
    test_linear_regression_convergence()
    test_xor_nonlinear_convergence()
