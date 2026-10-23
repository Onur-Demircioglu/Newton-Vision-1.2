"""Doğruluk Testi Suite (v0.2): C++ pybind11 çekirdeği vs. Saf Python referansı.

Bu test C++ im2col+GEMM conv2d ve yeni katmanların (AvgPool2d, LeakyReLU, Sigmoid, Tanh, BatchNorm2d)
saf Python NumPy referansı ile birebir aynı (np.allclose) olduğunu doğrular.
"""

from __future__ import annotations
import sys
import os
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python_ref"))

import layers as ref
import newton_vision as nv
import newton_vision._core as _core


def test_conv2d_correctness():
    print("--- Conv2D (im2col + GEMM) Doğruluk Testleri ---")
    np.random.seed(42)

    configs = [
        (1, 1, 3, 3, 1, 2, 2, 1, 0, False),
        (2, 3, 8, 8, 5, 3, 3, 2, 1, True),
        (4, 8, 14, 14, 16, 3, 3, 1, 1, True),
        (2, 4, 16, 16, 8, 5, 5, 2, 2, False),
    ]

    for i, (N, C_in, H, W, C_out, kH, kW, stride, padding, use_bias) in enumerate(configs, 1):
        x = np.random.randn(N, C_in, H, W).astype(np.float32)
        w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
        bias = np.random.randn(C_out).astype(np.float32) if use_bias else None

        ref_out = ref.conv2d(x, w, bias=bias, stride=stride, padding=padding)
        cpp_out = _core.conv2d(x, w, bias=bias, stride=stride, padding=padding)

        assert ref_out.shape == cpp_out.shape, f"Conv config {i} boyut uyuşmazlığı"
        assert np.allclose(ref_out, cpp_out, rtol=1e-5, atol=1e-5), f"Conv config {i} değer uyuşmazlığı!"
        print(f"  ✓ Conv config {i} geçti (im2col+GEMM): shape={cpp_out.shape}")


def test_activations_correctness():
    print("--- Aktivasyon Katmanları (ReLU, LeakyReLU, Sigmoid, Tanh) Testleri ---")
    np.random.seed(42)
    x = np.random.randn(2, 8, 16, 16).astype(np.float32)

    # 1. ReLU
    assert np.allclose(ref.relu(x), _core.relu(x), rtol=1e-5, atol=1e-5)
    print("  ✓ ReLU geçti")

    # 2. LeakyReLU
    ref_lrelu = np.where(x > 0, x, x * 0.01).astype(np.float32)
    assert np.allclose(ref_lrelu, _core.leaky_relu(x, 0.01), rtol=1e-5, atol=1e-5)
    print("  ✓ LeakyReLU geçti")

    # 3. Sigmoid
    ref_sig = (1.0 / (1.0 + np.exp(-x))).astype(np.float32)
    assert np.allclose(ref_sig, _core.sigmoid(x), rtol=1e-5, atol=1e-5)
    print("  ✓ Sigmoid geçti")

    # 4. Tanh
    ref_tanh = np.tanh(x).astype(np.float32)
    assert np.allclose(ref_tanh, _core.tanh(x), rtol=1e-5, atol=1e-5)
    print("  ✓ Tanh geçti")


def test_pooling_correctness():
    print("--- Pooling Katmanları (MaxPool2D, AvgPool2D) Testleri ---")
    np.random.seed(42)
    x = np.random.randn(2, 4, 8, 8).astype(np.float32)

    # 1. MaxPool2d
    ref_max = ref.maxpool2d(x, kernel_size=2, stride=2)
    cpp_max = _core.maxpool2d(x, kernel_size=2, stride=2)
    assert np.allclose(ref_max, cpp_max, rtol=1e-5, atol=1e-5)
    print("  ✓ MaxPool2d geçti")

    # 2. AvgPool2d
    # Ref AvgPool:
    N, C, H, W = x.shape
    ref_avg = np.empty((N, C, H // 2, W // 2), dtype=np.float32)
    for i in range(H // 2):
        for j in range(W // 2):
            ref_avg[:, :, i, j] = np.mean(x[:, :, i*2:i*2+2, j*2:j*2+2], axis=(2, 3))
    cpp_avg = _core.avgpool2d(x, kernel_size=2, stride=2)
    assert np.allclose(ref_avg, cpp_avg, rtol=1e-5, atol=1e-5)
    print("  ✓ AvgPool2d geçti")


def test_batchnorm_correctness():
    print("--- BatchNorm2D Doğruluk Testi ---")
    np.random.seed(42)
    N, C, H, W = 2, 4, 8, 8
    x = np.random.randn(N, C, H, W).astype(np.float32)
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)
    rm = np.random.randn(C).astype(np.float32)
    rv = np.abs(np.random.randn(C)).astype(np.float32) + 0.1
    eps = 1e-5

    # Ref BatchNorm (eval mode)
    ref_bn = (x - rm.reshape(1, C, 1, 1)) / np.sqrt(rv.reshape(1, C, 1, 1) + eps) * gamma.reshape(1, C, 1, 1) + beta.reshape(1, C, 1, 1)
    cpp_bn = _core.batchnorm2d(x, gamma, beta, rm, rv, eps)

    assert np.allclose(ref_bn, cpp_bn, rtol=1e-5, atol=1e-5)
    print("  ✓ BatchNorm2d geçti")


def test_optics_module():
    print("--- Optics Modül Alias Testi ---")
    conv = nv.optics.Conv2d(3, 16, 3)
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    out = conv(x)
    assert out.shape == (1, 16, 30, 30)
    print("  ✓ nv.optics.Conv2d nesne oluşturma ve forward işlemi geçti")


def main():
    print("==========================================")
    print("newton-vision v0.2: C++ Core & Extended Layers Parite Testi")
    print("==========================================")
    test_conv2d_correctness()
    test_activations_correctness()
    test_pooling_correctness()
    test_batchnorm_correctness()
    test_optics_module()
    print("\n✅ TÜM v0.2 DOĞRULUK TESTLERİ BAŞARIYLA GEÇTİ!")


def load_tests(loader, tests, pattern):
    """`python -m unittest discover` modül düzeyindeki test_* fonksiyonlarını da çalıştırsın diye."""
    funcs = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in funcs)


if __name__ == "__main__":
    main()
