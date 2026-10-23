"""apps/pytorch_int8_benchmark.py - PyTorch Agirlik Paritesi ve INT8 Kuantizasyon Karsilastirmasi.

Bu arac:
1. Gercek PyTorch uzerinde bir CNN modeli kurar ve state_dict cikarir.
2. newton-vision uzerinde ayni modeli olusturup agirliklari aktarir.
3. PyTorch vs newton-vision numerik paritesini (Parity & Cosine Similarity) dogrular.
4. Modeli INT8'e kuantize ederek %75 bellek kuculmesini ve cikarim hizini (FPS) olcer.

Kullanim:
    python apps/pytorch_int8_benchmark.py
"""

from __future__ import annotations
import os
import sys
import time
import numpy as np
import torch
import torch.nn as tnn
import newton_vision as nv


class PyTorchCNN(tnn.Module):
    """PyTorch Referans Modeli."""
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
        h1 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h2 = self.pool2(self.relu2(self.bn2(self.conv2(h1))))
        flat = h2.view(h2.size(0), -1)
        dense = self.relu3(self.fc1(flat))
        return self.classifier(dense)


class NewtonVisionCNN(nv.Module):
    """newton-vision Birebir Karsiligi Model."""
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
        h1 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h2 = self.pool2(self.relu2(self.bn2(self.conv2(h1))))
        N = h2.data.shape[0]
        flat = h2.reshape(N, 32 * 8 * 8)
        dense = self.relu3(self.fc1(flat))
        return self.classifier(dense)


def run_benchmark():
    print("\n" + "=" * 70)
    print(" PYTORCH AGIRLIK AKTARIMI & INT8 KUANTIZASYON BENCHMARK RAPORU")
    print("=" * 70)

    # 1. Deterministik PyTorch Modeli ve Agirliklari
    torch.manual_seed(42)
    pt_model = PyTorchCNN().eval()

    # PyTorch state dict'i numpy dizilerine cevir
    pt_state = {k: v.detach().cpu().numpy() for k, v in pt_model.state_dict().items()}
    print(f"[1/4] PyTorch {torch.__version__} modeli olusturuldu ({len(pt_state)} tensör).")

    # 2. newton-vision Modelini Olustur ve PyTorch Agirliklarini Yükle
    nv_model = NewtonVisionCNN().eval()
    nv.load_state_dict(nv_model, pt_state)
    print("[2/4] PyTorch agirliklari newton-vision modeline aktarildi.")

    # 3. Ortak Test Girdisi (1, 3, 32, 32)
    np.random.seed(42)
    input_np = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_torch = torch.from_numpy(input_np)
    input_nv = nv.Tensor(input_np)

    # Cikarim (PyTorch)
    with torch.no_grad():
        pt_out = pt_model(input_torch).numpy()

    # Cikarim (newton-vision FP32)
    with nv.no_grad():
        nv_out = nv_model(input_nv).data

    # Parite Olcumu
    max_diff = float(np.max(np.abs(pt_out - nv_out)))
    mean_diff = float(np.mean(np.abs(pt_out - nv_out)))
    # Cosine Similarity
    norm_pt = np.linalg.norm(pt_out)
    norm_nv = np.linalg.norm(nv_out)
    cosine_sim = float(np.dot(pt_out.flatten(), nv_out.flatten()) / (norm_pt * norm_nv))

    print("\n" + "-" * 70)
    print(" [BOLUM A: PYTORCH VS NEWTON-VISION MATEMATIKSEL PARITE]")
    print("-" * 70)
    print(f"  Maksimum Mutlak Fark (Max Abs Diff) : {max_diff:.8f}")
    print(f"  Ortalama Fark (Mean Abs Diff)        : {mean_diff:.8f}")
    print(f"  Kosinus Benzerligi (Cosine Sim)      : {cosine_sim * 100.0:.6f}%")
    print(f"  PyTorch Logits Ciktisi              : {np.round(pt_out[0, :5], 4)} ...")
    print(f"  newton-vision Logits Ciktisi        : {np.round(nv_out[0, :5], 4)} ...")

    if cosine_sim > 0.9999:
        print(f"  >>> [PARITE DOGRULANDI] Test edilen model ve giris uzerinde, newton-vision ciktisi PyTorch referansina floating-point toleransi ({max_diff:.1e}) dahilinde parite gostermistir.")
    else:
        print("  >>> [UYARI] Numerik fark tespit edildi.")

    # 4. INT8 Kuantizasyon ve Bellek Olcumu
    print("\n" + "-" * 70)
    print(" [BOLUM B: INT8 KUANTIZASYON & PARAMETRE BELLEK TASARRUFU]")
    print("-" * 70)

    # INT8 Modeli ayri olusturalim ki FP32 ile karsilastirabilelim
    nv_model_int8 = NewtonVisionCNN().eval()
    nv.load_state_dict(nv_model_int8, pt_state)

    stats = nv.quantize_model(nv_model_int8)

    print(f"  Kuantize Edilen Parametre Sayisi         : {stats['total_quantized_params']:,}")
    print(f"  Orijinal FP32 Parametre Bellek Boyutu    : {stats['fp32_memory_kb']:.2f} KB")
    print(f"  INT8 Kuantize Parametre Bellek Boyutu    : {stats['int8_memory_kb']:.2f} KB")
    print(f"  Kuantize Parametre Bellek Tasarrufu      : {stats['fp32_memory_kb'] - stats['int8_memory_kb']:.2f} KB  (%{stats['memory_reduction_percent']:.2f} tasarruf)")

    # 5. Hız ve Gecikme Benchmark'ı (100 Cikarim)
    print("\n" + "-" * 70)
    print(" [BOLUM C: HIZ & GECIKME KIYASLAMASI (100 ITERASYON)]")
    print("-" * 70)

    # FP32 Hız
    start = time.perf_counter()
    with nv.no_grad():
        for _ in range(100):
            _ = nv_model(input_nv)
    fp32_total_sec = time.perf_counter() - start
    fp32_latency = (fp32_total_sec / 100.0) * 1000.0
    fp32_fps = 100.0 / fp32_total_sec

    # INT8 Hız
    start = time.perf_counter()
    with nv.no_grad():
        for _ in range(100):
            _ = nv_model_int8(input_nv)
    int8_total_sec = time.perf_counter() - start
    int8_latency = (int8_total_sec / 100.0) * 1000.0
    int8_fps = 100.0 / int8_total_sec

    print(f"  FP32 Cikarim Gecikmesi : {fp32_latency:6.3f} ms / kare  |  Hiz: {fp32_fps:7.1f} FPS")
    print(f"  INT8 Cikarim Gecikmesi : {int8_latency:6.3f} ms / kare  |  Hiz: {int8_fps:7.1f} FPS")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
