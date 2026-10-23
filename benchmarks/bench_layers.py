"""Benchmark: C++ pybind11 Çekirdeği vs Saf Python Referansı.

Performans ve hız karşılaştırmasını (Speedup factor) ölçer.
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python_ref"))

import layers as ref
import newton_vision._core as _core


def benchmark_conv2d():
    print("\n[BENCHMARK] Conv2D (N=4, C_in=16, H=64, W=64 -> C_out=32, 3x3)")
    np.random.seed(42)

    x = np.random.randn(4, 16, 64, 64).astype(np.float32)
    w = np.random.randn(32, 16, 3, 3).astype(np.float32)
    bias = np.random.randn(32).astype(np.float32)

    warmup = 3
    iters = 15

    # Warmup
    for _ in range(warmup):
        _ = ref.conv2d(x, w, bias=bias, stride=1, padding=1)
        _ = _core.conv2d(x, w, bias=bias, stride=1, padding=1)

    # Benchmark Ref
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = ref.conv2d(x, w, bias=bias, stride=1, padding=1)
    t_ref = (time.perf_counter() - t0) / iters

    # Benchmark C++
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _core.conv2d(x, w, bias=bias, stride=1, padding=1)
    t_cpp = (time.perf_counter() - t0) / iters

    speedup = t_ref / t_cpp if t_cpp > 0 else 0
    print(f"  Python Ref : {t_ref * 1000:.2f} ms / iter")
    print(f"  C++ Naive  : {t_cpp * 1000:.2f} ms / iter")
    print(f"  🚀 Speedup : {speedup:.2f}x daha hızlı!")


def benchmark_relu():
    print("\n[BENCHMARK] ReLU (16 x 64 x 128 x 128)")
    np.random.seed(42)

    x = np.random.randn(16, 64, 128, 128).astype(np.float32)
    warmup = 5
    iters = 30

    for _ in range(warmup):
        _ = ref.relu(x)
        _ = _core.relu(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = ref.relu(x)
    t_ref = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _core.relu(x)
    t_cpp = (time.perf_counter() - t0) / iters

    speedup = t_ref / t_cpp if t_cpp > 0 else 0
    print(f"  Python Ref : {t_ref * 1000:.2f} ms / iter")
    print(f"  C++ Core   : {t_cpp * 1000:.2f} ms / iter")
    print(f"  🚀 Speedup : {speedup:.2f}x daha hızlı!")


def benchmark_maxpool2d():
    print("\n[BENCHMARK] MaxPool2D (N=8, C=32, H=64, W=64, k=2, s=2)")
    np.random.seed(42)

    x = np.random.randn(8, 32, 64, 64).astype(np.float32)
    warmup = 3
    iters = 20

    for _ in range(warmup):
        _ = ref.maxpool2d(x, kernel_size=2, stride=2)
        _ = _core.maxpool2d(x, kernel_size=2, stride=2)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = ref.maxpool2d(x, kernel_size=2, stride=2)
    t_ref = (time.perf_counter() - t0) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = _core.maxpool2d(x, kernel_size=2, stride=2)
    t_cpp = (time.perf_counter() - t0) / iters

    speedup = t_ref / t_cpp if t_cpp > 0 else 0
    print(f"  Python Ref : {t_ref * 1000:.2f} ms / iter")
    print(f"  C++ Naive  : {t_cpp * 1000:.2f} ms / iter")
    print(f"  🚀 Speedup : {speedup:.2f}x daha hızlı!")


def main():
    print("==========================================")
    print("newton-vision: Performans Benchmark")
    print("==========================================")
    benchmark_conv2d()
    benchmark_relu()
    benchmark_maxpool2d()


if __name__ == "__main__":
    main()
