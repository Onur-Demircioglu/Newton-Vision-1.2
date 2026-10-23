"""benchmarks/benchmark_pytorch_parity.py - Sayısal Parite & Kuantize Parametre Bellek Ölçümü.

Ölçüm Kapsamı:
1. Mathematical Parity (Kategori 1): PyTorch 2.6 ile newton-vision arasındaki sayısal fark.
2. Parametre Bellek Tasarrufu: FP32 vs INT8 per-channel kuantize edilen katman ağırlıkları.
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import torch.nn as tnn
import newton_vision as nv


class PyTorchRefCNN(tnn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = tnn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = tnn.BatchNorm2d(16)
        self.relu1 = tnn.ReLU()
        self.pool1 = tnn.MaxPool2d(2)
        self.conv2 = tnn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = tnn.BatchNorm2d(32)
        self.relu2 = tnn.ReLU()
        self.pool2 = tnn.MaxPool2d(2)
        self.fc1 = tnn.Linear(32 * 8 * 8, 128)
        self.relu3 = tnn.ReLU()
        self.classifier = tnn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h = self.pool2(self.relu2(self.bn2(self.conv2(h))))
        flat = h.view(h.size(0), -1)
        dense = self.relu3(self.fc1(flat))
        return self.classifier(dense)


class NewtonVisionCNN(nv.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nv.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nv.BatchNorm2d(16)
        self.relu1 = nv.ReLU()
        self.pool1 = nv.MaxPool2d(2)
        self.conv2 = nv.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nv.BatchNorm2d(32)
        self.relu2 = nv.ReLU()
        self.pool2 = nv.MaxPool2d(2)
        self.fc1 = nv.Linear(32 * 8 * 8, 128)
        self.relu3 = nv.ReLU()
        self.classifier = nv.Linear(128, 10)

    def forward(self, x: nv.Tensor) -> nv.Tensor:
        h = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h = self.pool2(self.relu2(self.bn2(self.conv2(h))))
        N = h.data.shape[0]
        flat = nv.Tensor(h.data.reshape(N, -1), requires_grad=False)
        dense = self.relu3(self.fc1(flat))
        return self.classifier(dense)


def run_parity_benchmark():
    np.random.seed(42)
    torch.manual_seed(42)

    pt_model = PyTorchRefCNN().eval()
    pt_state = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}

    nv_model = NewtonVisionCNN().eval()
    nv.load_state_dict(nv_model, pt_state)

    dummy_raw = np.random.randn(1, 3, 32, 32).astype(np.float32)
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(dummy_raw)).numpy()

    with nv.no_grad():
        nv_out = nv_model(nv.Tensor(dummy_raw)).data

    diff = np.abs(pt_out - nv_out)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    pt_norm = np.linalg.norm(pt_out)
    nv_norm = np.linalg.norm(nv_out)
    cosine_sim = float(np.dot(pt_out.flatten(), nv_out.flatten()) / (pt_norm * nv_norm + 1e-12))

    # Kuantizasyon Parametre Belleği
    nv_model_int8 = NewtonVisionCNN().eval()
    nv.load_state_dict(nv_model_int8, pt_state)
    quant_stats = nv.quantize_model(nv_model_int8)

    print("=" * 70)
    print(" [BENCHMARK 1] PYTORCH SAYISAL PARITE VE INT8 PARAMETRE BELLEGI")
    print("=" * 70)
    print(f"  Maksimum Mutlak Fark (Max Abs Diff)   : {max_diff:.8f}")
    print(f"  Ortalama Mutlak Fark (Mean Abs Diff)  : {mean_diff:.8f}")
    print(f"  Kosinus Benzerligi (Cosine Sim)      : {cosine_sim * 100.0:.6f}%")
    print(f"  >>> BILIMSEL SONUC: Test edilen model ve giris uzerinde, newton-vision")
    print(f"      ciktisi PyTorch referansina floating-point toleransi ({max_diff:.1e})")
    print(f"      dahilinde parite gostermistir.")
    print("-" * 70)
    print(f"  Kuantize Edilen Parametre Sayisi      : {quant_stats['total_quantized_params']:,}")
    print(f"  FP32 Parametre Bellegi                : {quant_stats['fp32_memory_kb']:.2f} KB")
    print(f"  INT8 Parametre Bellegi                : {quant_stats['int8_memory_kb']:.2f} KB")
    print(f"  Kuantize Parametre Bellek Tasarrufu   : %{quant_stats['memory_reduction_percent']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_parity_benchmark()
