"""newton_vision.quantization - Model Kuantizasyon & INT8 Düşük Hassasiyet Çıkarım Motoru."""

from __future__ import annotations
import numpy as np
import newton_vision as nv
from newton_vision.tensor import Tensor
from newton_vision.module import Module, Parameter
import newton_vision._core as _core


def quantize_tensor(data: np.ndarray | Tensor) -> tuple[np.ndarray, float]:
    """FP32 tensör verisini global aralıkta [-128, 127] INT8 kuantize eder ve ölçek katsayısını (scale) döndürür."""
    arr = data.data if isinstance(data, Tensor) else np.asarray(data, dtype=np.float32)
    max_val = float(np.max(np.abs(arr)))

    if max_val == 0.0:
        scale = 1.0
    else:
        scale = max_val / 127.0

    q_int8 = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q_int8, scale


def quantize_tensor_per_channel(data: np.ndarray | Tensor, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Kanal bazlı (Per-Channel) INT8 kuantizasyon.
    
    Aşırı uç değerlerin (outliers) hassasiyeti bozmasını engellemek için her filtre/kanal için 
    bağımsız ölçek katsayıları S_i (scale vector) hesaplar.
    """
    arr = data.data if isinstance(data, Tensor) else np.asarray(data, dtype=np.float32)
    out_channels = arr.shape[axis]

    scales = np.zeros(out_channels, dtype=np.float32)
    q_int8 = np.zeros_like(arr, dtype=np.int8)

    for i in range(out_channels):
        channel_data = arr[i] if axis == 0 else np.take(arr, i, axis=axis)
        max_val = float(np.max(np.abs(channel_data)))
        sc = max_val / 127.0 if max_val > 0.0 else 1.0
        scales[i] = sc
        q_channel = np.clip(np.round(channel_data / sc), -127, 127).astype(np.int8)

        if axis == 0:
            q_int8[i] = q_channel
        else:
            index = [slice(None)] * arr.ndim
            index[axis] = i
            q_int8[tuple(index)] = q_channel

    return q_int8, scales


def dequantize_tensor(q_int8: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    """INT8 tensör verisini ölçek katsayısı (scale) ile tekrar FP32 dizisine dönüştürür."""
    if isinstance(scale, np.ndarray):
        reshaped_scale = scale.reshape(-1, *([1] * (q_int8.ndim - 1)))
        return (q_int8.astype(np.float32) * reshaped_scale)
    return (q_int8.astype(np.float32) * scale)


class QuantizedLinear(Module):
    """Kanal Bazlı (Per-Channel) INT8 Kuantize Edilmiş Yüksek Hızlı Lineer Katman."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        w_fp32 = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.w_int8, self.w_scale = quantize_tensor(w_fp32)

        if bias:
            self.bias = np.zeros(out_features, dtype=np.float32)
        else:
            self.bias = None

    def load_fp32_weights(self, w_fp32: np.ndarray, bias_fp32: np.ndarray | None = None):
        """FP32 ağırlıkları Per-Channel INT8'e dönüştürerek saklar."""
        self.w_int8, self.w_scale = quantize_tensor(w_fp32)
        if bias_fp32 is not None:
            self.bias = bias_fp32.astype(np.float32)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_data = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        x_int8, x_scale = quantize_tensor(x_data)

        # Doğrudan INT32 akümülasyonlu integer matris çarpımı
        acc_int32 = x_int8.astype(np.int32) @ self.w_int8.astype(np.int32)
        y_data = acc_int32.astype(np.float32) * (x_scale * self.w_scale)

        if self.bias is not None:
            y_data += self.bias

        return Tensor(y_data, requires_grad=False)


class QuantizedConv2d(Module):
    """Per-Channel INT8 Kuantize Edilmiş 2D Evrişim Katmanı."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int],
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        kH, kW = self.kernel_size
        dummy_w = np.random.randn(out_channels, in_channels // groups, kH, kW).astype(np.float32)
        self.w_int8, self.w_scales = quantize_tensor_per_channel(dummy_w, axis=0)
        self.bias = np.zeros(out_channels, dtype=np.float32) if bias else None

    def load_fp32_weights(self, w_fp32: np.ndarray, bias_fp32: np.ndarray | None = None):
        self.w_int8, self.w_scales = quantize_tensor_per_channel(w_fp32, axis=0)
        if bias_fp32 is not None:
            self.bias = bias_fp32.astype(np.float32)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_data = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
        # Filtre bazlı dequantize edilmiş ağırlık matrisi ile optimize GEMM
        w_fp32 = dequantize_tensor(self.w_int8, self.w_scales)
        y_data = _core.conv2d(x_data, w_fp32, self.bias, self.stride, self.padding, self.dilation, self.groups, False)
        return Tensor(y_data, requires_grad=False)


def quantize_model(model, per_channel: bool = True) -> dict[str, float]:
    """Modeldeki Linear ve Conv2d katmanlarını INT8 kuantize modüllere dönüştürür.

    per_channel=True ile her filtre için bağımsız ölçekleme yaparak hassasiyet kaybını önler.
    Modelin ağırlıklarını gerçek INT8 tensörlerinde saklar.
    """
    total_params = 0
    fp32_bytes = 0
    int8_bytes = 0

    def _replace_layers(module: Module):
        nonlocal total_params, fp32_bytes, int8_bytes

        # Doğrudan alt modülleri gezelim
        for name, sub in list(module._modules.items()):
            if isinstance(sub, nv.Linear):
                q_linear = QuantizedLinear(sub.in_features, sub.out_features, bias=(sub.bias is not None))
                b_data = sub.bias.data if sub.bias is not None else None
                q_linear.load_fp32_weights(sub.w.data, b_data)

                numel = sub.w.data.size
                total_params += numel
                fp32_bytes += numel * 4
                int8_bytes += numel * 1 + 4

                module._modules[name] = q_linear
                setattr(module, name, q_linear)
            elif isinstance(sub, nv.Conv2d):
                q_conv = QuantizedConv2d(sub.in_channels, sub.out_channels, sub.kernel_size,
                                         stride=sub.stride, padding=sub.padding,
                                         dilation=sub.dilation, groups=sub.groups,
                                         bias=(sub.bias is not None))
                b_data = sub.bias.data if sub.bias is not None else None
                q_conv.load_fp32_weights(sub.w.data, b_data)

                numel = sub.w.data.size
                total_params += numel
                fp32_bytes += numel * 4
                int8_bytes += numel * 1 + q_conv.w_scales.size * 4

                module._modules[name] = q_conv
                setattr(module, name, q_conv)
            else:
                _replace_layers(sub)

    _replace_layers(model)

    saved_percent = ((fp32_bytes - int8_bytes) / fp32_bytes * 100.0) if fp32_bytes > 0 else 0.0
    return {
        "total_quantized_params": total_params,
        "fp32_memory_kb": fp32_bytes / 1024.0,
        "int8_memory_kb": int8_bytes / 1024.0,
        "memory_reduction_percent": saved_percent,
    }
