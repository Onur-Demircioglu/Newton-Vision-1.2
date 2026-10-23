"""Autograd & Gradyan Doğrulama Testi (v0.3): Fluxion & Gravity.

Analitik gradyanları Sayısal Türev (Sonlu Farklar: (f(x+eps)-f(x-eps))/(2*eps)) ile doğrular
ve uçtan uca mini CNN eğitimi gerçekleştirir (Loss -> 0).
"""

from __future__ import annotations
import unittest
import numpy as np
import newton_vision as nv


def test_numerical_gradient_conv2d():
    print("--- Conv2D Sayısal Gradyan Doğrulaması (Finite Differences) ---")
    np.random.seed(42)

    conv = nv.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=1)
    x = nv.Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32), requires_grad=True)

    y = conv(x)
    loss = (y * y).sum()
    loss.backward()

    analyt_grad_w = conv.w.grad.copy()

    eps = 1e-3
    num_grad_w = np.zeros_like(conv.w.data)

    it = np.nditer(conv.w.data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig_val = conv.w.data[idx]

        conv.w.data[idx] = orig_val + eps
        y_plus = conv(x)
        l_plus = (y_plus * y_plus).sum().data.item()

        conv.w.data[idx] = orig_val - eps
        y_minus = conv(x)
        l_minus = (y_minus * y_minus).sum().data.item()

        conv.w.data[idx] = orig_val
        num_grad_w[idx] = (l_plus - l_minus) / (2.0 * eps)
        it.iternext()

    diff = np.max(np.abs(analyt_grad_w - num_grad_w))
    print(f"  ✓ Conv2D W Analitik vs Sayısal Gradyan Maksimum Fark: {diff:.6f}")
    assert diff < 1e-2, f"Conv2D gradyan hatası sayısal eşiği aştı: {diff}"


def test_end_to_end_training():
    print("\n--- Uçtan Uca Mini CNN Eğitimi (Loss Düşüş Testi) ---")
    np.random.seed(42)

    x = np.random.randn(1, 2, 4, 4).astype(np.float32)
    target = np.random.randn(1, 2, 4, 4).astype(np.float32)

    conv = nv.Conv2d(2, 2, 3, padding=1)
    criterion = nv.gravity.MSELoss()
    optimizer = nv.gravity.Adam(conv.parameters(), lr=0.05)

    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(100):
        optimizer.zero_grad()
        out = conv(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.data.item()
        final_loss = loss.data.item()

    print(f"  ✓ Başlangıç Kaybı (Epoch 1)   : {initial_loss:.6f}")
    print(f"  ✓ Bitiş Kaybı     (Epoch 100) : {final_loss:.6f}")
    assert final_loss < initial_loss * 0.05, "Model kayıpları beklenen seviyede düşmedi!"
    print("  🚀 Model başarıyla eğitildi ve kayıp sıfıra yaklaştı (%95+ Düşüş)!")


def main():
    print("==========================================")
    print("newton-vision v0.3: Fluxion Autograd & Gradyan Testleri")
    print("==========================================")
    test_numerical_gradient_conv2d()
    test_end_to_end_training()
    print("\n✅ TÜM AUTOGRAD VE EĞİTİM TESTLERİ GEÇTİ!")


def load_tests(loader, tests, pattern):
    """`python -m unittest discover` modül düzeyindeki test_* fonksiyonlarını da çalıştırsın diye."""
    funcs = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in funcs)


if __name__ == "__main__":
    main()
