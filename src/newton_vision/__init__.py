"""
[NEWTON-VISION] High-Performance Computer Vision & Deep Learning Framework
=============================================================================
Yuksek Performansli C++17 & CUDA Ivmelendirmeli Derin Ogrenme Kutuphanesi

[EN] newton-vision is a lightweight C++17 & CUDA accelerated deep learning framework 
     with reverse-mode Autograd, PyTorch state_dict parity, AVX2 SIMD, and INT8 Quantization.
[TR] newton-vision; C++17 ve CUDA destekli, Otomatik Turev (Autograd) motoru, PyTorch 
     agirlik uyumlulugu, AVX2 SIMD ve INT8 Kuantizasyon iceren derin ogrenme kutuphanesidir.

MODULE OVERVIEW / MODUL HARITASI (Newton System)
------------------------------------------------
* newton_vision.optics       : Gorsel Katmanlar (Conv2d, Linear, BatchNorm2d, LayerNorm...)
* newton_vision.fluxion      : Turev Motoru (Tensor, backward, .cuda(), .cpu(), .to(device))
* newton_vision.gravity      : Kayip & Optimizator (MSELoss, CrossEntropyLoss, SGD, Adam)
* newton_vision.principia    : Cihaz Yonetimi (is_cuda_available, get_device, set_device)
* newton_vision.models       : Model Zoo (ResNet18, MobileNetV2, VisionTransformer...)
* newton_vision.quantization : INT8 Kuantizasyon & SIMD Engine (quantize_model...)
* newton_vision.weights      : Agirlik Aktarici / Weight Importer (load_state_dict)
* newton_vision.jit          : Canli Sunucu & C++ Serializer (save_model, load_model)

QUICK EXAMPLE / HIZLI ÖRNEK
---------------------------
>>> import newton_vision as nv
>>> import numpy as np
>>> x = nv.Tensor(np.random.randn(2, 3, 224, 224), requires_grad=True)
>>> model = nv.models.vit_tiny(num_classes=1000)
>>> logits = model(x)
>>> print(logits.shape)
(2, 1000)

Detaylı doküman için `help(newton_vision.optics)` veya `help(newton_vision.Tensor)` komutunu kullanabilirsiniz.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path


def _add_dev_core_path() -> None:
    package_dir = Path(__file__).resolve().parent
    project_root = package_dir.parents[1]
    build_dir = project_root / "build"

    if not build_dir.exists():
        return

    for core_path in build_dir.glob("*/_core*.pyd"):
        core_dir = str(core_path.parent)
        if core_dir not in __path__:
            __path__.append(core_dir)
        if core_dir not in sys.path:
            sys.path.insert(0, core_dir)
        break

# On Windows with MinGW, attempt adding MSYS2/MinGW DLL directory if needed
if sys.platform == "win32":
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.exists(mingw_bin):
        try:
            os.add_dll_directory(mingw_bin)
        except Exception:
            pass
    # MSVC + CUDA build'i cudart64_*.dll'e ihtiyaç duyar; Python 3.8+ bağımlı DLL'ler için PATH'e bakmaz
    _cuda_bin = os.path.join(os.environ.get("CUDA_PATH", ""), "bin")
    if os.environ.get("CUDA_PATH") and os.path.isdir(_cuda_bin):
        os.add_dll_directory(_cuda_bin)

_add_dev_core_path()

from newton_vision.tensor import Tensor, no_grad, enable_grad, is_grad_enabled
from newton_vision.module import Module, Parameter
from newton_vision.containers import Sequential, ModuleList
from newton_vision.data import Dataset, TensorDataset, DataLoader, ImageFolder, CSVDataset
from newton_vision.layers import (
    Conv2d,
    Linear,
    LazyLinear,
    ReLU,
    LeakyReLU,
    Sigmoid,
    Tanh,
    GELU,
    MaxPool2d,
    AvgPool2d,
    BatchNorm2d,
    LayerNorm,
    MultiHeadAttention,
    Dropout,
    Flatten,
    Embedding,
    BatchNorm1d,
    Conv1d,
    Upsample,
)
from newton_vision.gravity import CrossEntropyLoss, MSELoss, L1Loss, BCELoss, BCEWithLogitsLoss, SGD, Adam, AdamW, StepLR, CosineAnnealingLR, clip_grad_norm_
from newton_vision import optics, fluxion, gravity, principia, models, weights, jit, quantization, data, containers, transforms, text
from newton_vision.weights import load_state_dict
from newton_vision.jit import save_model, load_model, save_model_bin, load_model_bin, export_onnx, InferenceEngine
from newton_vision.quantization import (
    quantize_tensor,
    quantize_tensor_per_channel,
    dequantize_tensor,
    quantize_model,
    QuantizedLinear,
    QuantizedConv2d,
)

__version__ = "1.2.0"


def help(topic: str | None = None) -> None:
    """Interaktif Kullanim Kilavuzunu basar.
    
    Ornekler:
        nv.help()           # Genel kilavuz ve modul haritasi
        nv.help('layers')   # Katmanlar rehberi
        nv.help('train')    # Egitim dongusu sablonu
        nv.help('pytorch')  # PyTorch agirlik aktarimi
        nv.help('quant')    # INT8 kuantizasyon rehberi
        nv.help('apps')     # 3 somut uygulama rehberi
    """
    from newton_vision.__main__ import (
        print_general_help,
        print_layers_help,
        print_training_help,
        print_pytorch_help,
        print_quant_help,
        print_apps_help,
    )
    t = (topic or "").lower()
    if t in ("layers", "layer"):
        print_layers_help()
    elif t in ("training", "train"):
        print_training_help()
    elif t in ("pytorch", "weights"):
        print_pytorch_help()
    elif t in ("quant", "quantization", "int8"):
        print_quant_help()
    elif t in ("apps", "app"):
        print_apps_help()
    else:
        print_general_help()


__all__ = [
    "Tensor",
    "Module",
    "Parameter",
    "Sequential",
    "ModuleList",
    "Dataset",
    "TensorDataset",
    "DataLoader",
    "ImageFolder",
    "CSVDataset",
    "no_grad",
    "enable_grad",
    "is_grad_enabled",
    "Conv2d",
    "Linear",
    "LazyLinear",
    "ReLU",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "GELU",
    "MaxPool2d",
    "AvgPool2d",
    "BatchNorm2d",
    "LayerNorm",
    "MultiHeadAttention",
    "Dropout",
    "Flatten",
    "Embedding",
    "BatchNorm1d",
    "Conv1d",
    "Upsample",
    "CrossEntropyLoss",
    "MSELoss",
    "L1Loss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "SGD",
    "Adam",
    "AdamW",
    "StepLR",
    "CosineAnnealingLR",
    "clip_grad_norm_",
    "optics",
    "fluxion",
    "gravity",
    "principia",
    "models",
    "weights",
    "jit",
    "quantization",
    "data",
    "containers",
    "transforms",
    "text",
    "load_state_dict",
    "save_model",
    "load_model",
    "save_model_bin",
    "load_model_bin",
    "export_onnx",
    "InferenceEngine",
    "quantize_tensor",
    "quantize_tensor_per_channel",
    "dequantize_tensor",
    "quantize_model",
    "QuantizedLinear",
    "QuantizedConv2d",
    "help",
    "__version__",
]
