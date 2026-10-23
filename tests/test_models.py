"""Vision Model Zoo & Weight Importer Testi (tests/test_models.py).

ResNet-18 ve MobileNetV2 mimarilerinin newton-vision ile ileri besleme (forward pass)
çalışmasını ve PyTorch state_dict ağırlık aktarımı fonksiyonunu doğrular.
"""

from __future__ import annotations
import unittest
import numpy as np
import newton_vision as nv


def test_linear_layer():
    print("--- Linear Katmanı Testi ---")
    fc = nv.Linear(in_features=10, out_features=5)
    x = nv.Tensor(np.random.randn(2, 10).astype(np.float32), requires_grad=True)

    y = fc(x)
    assert y.shape == (2, 5), f"Linear çıktı boyutu yanlış: {y.shape}"

    loss = y.sum()
    loss.backward()
    assert x.grad is not None and fc.w.grad is not None, "Linear autograd gradyanı hesaplanamadı!"
    print(f"  [OK] Linear katmani forward/backward testi gecti: shape={y.shape}")


def test_weight_importer():
    print("--- PyTorch State Dict Ağırlık Aktarıcı Testi ---")
    conv = nv.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    fake_state_dict = {
        "weight": np.ones((16, 3, 3, 3), dtype=np.float32) * 2.5,
        "bias": np.ones((16,), dtype=np.float32) * 0.5
    }

    nv.load_state_dict(conv, fake_state_dict)

    assert np.allclose(conv.w.data, 2.5), "Agirlik kopyalamasi uyusmuyor!"
    assert np.allclose(conv.bias.data, 0.5), "Bias kopyalamasi uyusmuyor!"
    print("  [OK] PyTorch state_dict aktarimi basariyla dogrulandi")


def test_resnet18():
    print("--- ResNet-18 Model Zoo Forward Pass Testi ---")
    model = nv.models.resnet18(num_classes=1000)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    logits = model(x)
    assert logits.shape == (1, 1000), f"ResNet-18 cikti boyutu yanlis: {logits.shape}"
    print(f"  [OK] ResNet-18 model forward testi gecti: shape={logits.shape}")


def test_mobilenet_v2():
    print("--- MobileNetV2 Model Zoo Forward Pass Testi ---")
    model = nv.models.mobilenet_v2(num_classes=1000)
    x = np.random.randn(1, 3, 224, 224).astype(np.float32)

    logits = model(x)
    assert logits.shape == (1, 1000), f"MobileNetV2 cikti boyutu yanlis: {logits.shape}"
    print(f"  [OK] MobileNetV2 model forward testi gecti: shape={logits.shape}")


def main():
    print("==========================================")
    print("newton-vision v0.5: Vision Model Zoo & Weights Testleri")
    print("==========================================")
    test_linear_layer()
    test_weight_importer()
    test_resnet18()
    test_mobilenet_v2()
    print("\n[OK] TUM MODEL ZOO VE AGIRLIK AKTARIM TESTLERI GECTI!")


def load_tests(loader, tests, pattern):
    """`python -m unittest discover` modül düzeyindeki test_* fonksiyonlarını da çalıştırsın diye."""
    funcs = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in funcs)


if __name__ == "__main__":
    main()
