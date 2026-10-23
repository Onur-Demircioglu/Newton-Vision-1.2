"""newton-vision yüksek seviye katman API'si.

PyTorch/CNN tarzı Conv2d, Linear, ReLU, MaxPool2d, AvgPool2d, LeakyReLU, Sigmoid, Tanh ve BatchNorm2d.
"""

from __future__ import annotations
import numpy as np
import newton_vision._core as _core
from newton_vision.tensor import Tensor
from newton_vision.module import Module, Parameter


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: str = "cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device.lower()

        scale = np.sqrt(2.0 / in_features, dtype=np.float32)
        self.w = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * scale, device=self.device)
        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float32), device=self.device)
        else:
            self.bias = None

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        if x_tensor.data.shape[-1] != self.in_features:
            raise ValueError(
                f"[Linear Boyut Hatası] Giriş tensörünün son boyutu ({x_tensor.data.shape[-1]}) "
                f"katmanın in_features ({self.in_features}) değeriyle eşleşmiyor! "
                f"Alınan tensör şekli: {x_tensor.data.shape}"
            )
        logits_data = x_tensor.data @ self.w.data
        if self.bias is not None:
            logits_data += self.bias.data

        req_grad = x_tensor.requires_grad or self.w.requires_grad or (self.bias is not None and self.bias.requires_grad)
        children = [x_tensor, self.w]
        if self.bias is not None:
            children.append(self.bias)

        out = Tensor(logits_data, requires_grad=req_grad, device=x_tensor.device, _children=children)

        if req_grad:
            def _backward():
                if x_tensor.requires_grad:
                    gx = out.grad @ self.w.data.T
                    x_tensor._add_grad(gx)
                if self.w.requires_grad:
                    x_2d = x_tensor.data.reshape(-1, self.in_features)
                    g_2d = out.grad.reshape(-1, self.out_features)
                    gw = x_2d.T @ g_2d
                    self.w._add_grad(gw)
                if self.bias is not None and self.bias.requires_grad:
                    gb = np.sum(out.grad.reshape(-1, self.out_features), axis=0)
                    self.bias._add_grad(gb)

            out._backward = _backward

        return out


class LazyLinear(Module):
    """Giris boyutunu (in_features) ilk gelen tensorden dinamik olarak ogrenen katman.
    
    Gelistiricinin verideki sutun/oznitelik sayisini onceden bilmesine veya
    Conv2d cikisindaki carpimlari (C * H * W) elle hesaplamasina gerek birakmaz.
    
    Ornek:
        fc = nv.LazyLinear(out_features=10)
        # x boyutu (Batch, 12) ise fc otomatik Linear(12, 10) olur.
        # x boyutu (Batch, 1024) ise fc otomatik Linear(1024, 10) olur.
        out = fc(x)
    """
    def __init__(self, out_features: int, bias: bool = True, device: str = "cpu"):
        super().__init__()
        self.out_features = out_features
        self.bias_flag = bias
        self.device = device.lower()
        self.in_features = None
        self.w = None
        self.bias = None

    def initialize_parameters(self, in_features: int):
        self.in_features = in_features
        scale = np.sqrt(2.0 / in_features, dtype=np.float32)
        w_tensor = Parameter(np.random.randn(in_features, self.out_features).astype(np.float32) * scale, device=self.device)
        self.register_parameter("w", w_tensor)
        if self.bias_flag:
            b_tensor = Parameter(np.zeros(self.out_features, dtype=np.float32), device=self.device)
            self.register_parameter("bias", b_tensor)
        else:
            self.bias = None

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        if self.w is None:
            self.initialize_parameters(x_tensor.data.shape[-1])

        logits_data = x_tensor.data @ self.w.data
        if self.bias is not None:
            logits_data += self.bias.data

        req_grad = x_tensor.requires_grad or self.w.requires_grad or (self.bias is not None and self.bias.requires_grad)
        children = [x_tensor, self.w]
        if self.bias is not None:
            children.append(self.bias)

        out = Tensor(logits_data, requires_grad=req_grad, device=x_tensor.device, _children=children)

        if req_grad:
            def _backward():
                if x_tensor.requires_grad:
                    gx = out.grad @ self.w.data.T
                    x_tensor._add_grad(gx)
                if self.w.requires_grad:
                    x_2d = x_tensor.data.reshape(-1, self.in_features)
                    g_2d = out.grad.reshape(-1, self.out_features)
                    gw = x_2d.T @ g_2d
                    self.w._add_grad(gw)
                if self.bias is not None and self.bias.requires_grad:
                    gb = np.sum(out.grad.reshape(-1, self.out_features), axis=0)
                    self.bias._add_grad(gb)

            out._backward = _backward

        return out


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.device = device.lower()

        kH, kW = self.kernel_size
        scale = np.sqrt(2.0 / (in_channels * kH * kW / groups), dtype=np.float32)
        self.w = Parameter(np.random.randn(out_channels, in_channels // groups, kH, kW).astype(np.float32) * scale, device=self.device)
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32), device=self.device)
        else:
            self.bias = None

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        if x_tensor.data.ndim != 4:
            raise ValueError(
                f"[Conv2d Boyut Hatası] Giriş tensörü 4 boyutlu (Batch, Kanal, Yükseklik, Genişlik) olmalıdır! "
                f"Alınan tensör şekli: {x_tensor.data.shape}"
            )
        if x_tensor.data.shape[1] != self.in_channels:
            raise ValueError(
                f"[Conv2d Kanal Hatası] Giriş kanal sayısı ({x_tensor.data.shape[1]}) "
                f"katmanın in_channels ({self.in_channels}) değeriyle eşleşmiyor!"
            )
        b_data = self.bias.data if self.bias is not None else None
        use_cuda = (x_tensor.device == "cuda")

        y_data = _core.conv2d(x_tensor.data, self.w.data, b_data, self.stride, self.padding, self.dilation, self.groups, use_cuda)

        req_grad = x_tensor.requires_grad or self.w.requires_grad or (self.bias is not None and self.bias.requires_grad)
        children = [x_tensor, self.w]
        if self.bias is not None:
            children.append(self.bias)

        out = Tensor(y_data, requires_grad=req_grad, device=x_tensor.device, _children=children)

        if req_grad:
            def _backward():
                gx, gw, gb = _core.conv2d_backward(out.grad, x_tensor.data, self.w.data, self.bias is not None, self.stride, self.padding, self.dilation, self.groups)
                if x_tensor.requires_grad:
                    x_tensor._add_grad(gx)
                if self.w.requires_grad:
                    self.w._add_grad(gw)
                if self.bias is not None and self.bias.requires_grad and gb is not None:
                    self.bias._add_grad(gb)

            out._backward = _backward

        return out


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        use_cuda = (x_tensor.device == "cuda")
        y_data = _core.relu(x_tensor.data, use_cuda)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                gx = _core.relu_backward(out.grad, x_tensor.data)
                x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        y_data = _core.leaky_relu(x_tensor.data, self.negative_slope)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                gx = out.grad * np.where(x_tensor.data > 0, 1.0, self.negative_slope).astype(np.float32)
                x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        y_data = _core.sigmoid(x_tensor.data)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                x_tensor._add_grad(out.grad * y_data * (1.0 - y_data))

            out._backward = _backward

        return out


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        y_data = _core.tanh(x_tensor.data)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                x_tensor._add_grad(out.grad * (1.0 - y_data ** 2))

            out._backward = _backward

        return out


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        if padding < 0 or padding > kernel_size // 2:
            # PyTorch ile aynı kısıt: her pencerede en az bir gerçek eleman kalmalı
            raise ValueError(f"MaxPool2d padding 0 ile kernel_size // 2 arasında olmalı, alınan: {padding}")
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        use_cuda = (x_tensor.device == "cuda")
        p = self.padding
        # -inf dolgu: C++ çekirdeği katı '>' karşılaştırması yaptığı için dolgu asla maksimum seçilmez
        x_data = np.pad(x_tensor.data, ((0, 0), (0, 0), (p, p), (p, p)), constant_values=-np.inf) if p > 0 else x_tensor.data
        y_data = _core.maxpool2d(x_data, self.kernel_size, self.stride, use_cuda)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                gx = _core.maxpool2d_backward(out.grad, x_data, self.kernel_size, self.stride)
                if p > 0:
                    gx = gx[:, :, p:-p, p:-p]
                x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        y_data = _core.avgpool2d(x_tensor.data, self.kernel_size, self.stride)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                gx = np.zeros_like(x_tensor.data, dtype=np.float32)
                k = self.kernel_size
                s = self.stride
                _, _, H_out, W_out = out.grad.shape
                scale = 1.0 / float(k * k)
                for i in range(H_out):
                    hs = i * s
                    for j in range(W_out):
                        ws = j * s
                        gx[:, :, hs:hs + k, ws:ws + k] += out.grad[:, :, i:i + 1, j:j + 1] * scale
                x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class Dropout(Module):
    """Eğitim sırasında nöronların bir kısmını (p) rastgele sıfırlayan düzenlileştirme (Regularization) katmanı.

    model.train() modunda inverted dropout (1 / (1-p)) uygular.
    model.eval() modunda sıfır ek maliyetle doğrudan kimlik geçişi (identity pass) yapar.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout olasılığı [0, 1) aralığında olmalıdır, alınan: {p}")
        self.p = p

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        if not self.training or self.p == 0.0:
            return x_tensor

        keep_prob = 1.0 - self.p
        mask = (np.random.rand(*x_tensor.data.shape) < keep_prob).astype(np.float32) / keep_prob
        y_data = x_tensor.data * mask
        req_grad = x_tensor.requires_grad
        out = Tensor(y_data, requires_grad=req_grad, device=x_tensor.device, _children=(x_tensor,))

        if req_grad:
            def _backward():
                x_tensor._add_grad(out.grad * mask)
            out._backward = _backward

        return out


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, device: str = "cpu"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.device = device.lower()
        self.gamma = Parameter(np.ones(num_features, dtype=np.float32), device=self.device)
        self.beta = Parameter(np.zeros(num_features, dtype=np.float32), device=self.device)
        self.register_buffer("running_mean", np.zeros(num_features, dtype=np.float32))
        self.register_buffer("running_var", np.ones(num_features, dtype=np.float32))

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        if self.training:
            sample_count = x_tensor.data.shape[0] * x_tensor.data.shape[2] * x_tensor.data.shape[3]
            if sample_count <= 1:
                raise ValueError(
                    f"[BatchNorm2d] Egitim modunda kanal basina 1'den fazla deger gerekir "
                    f"(N*H*W = {sample_count}). Varyans tanimsiz."
                )
            mean = np.mean(x_tensor.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x_tensor.data, axis=(0, 2, 3), keepdims=True)  # biased: normalizasyon icin
            x_hat = (x_tensor.data - mean) / np.sqrt(var + self.eps)
            y_data = x_hat * self.gamma.data.reshape(1, -1, 1, 1) + self.beta.data.reshape(1, -1, 1, 1)

            unbiased_var = var.reshape(-1) * (sample_count / (sample_count - 1))  # running icin (PyTorch paritesi)
            self.running_mean = (
                (1.0 - self.momentum) * self.running_mean +
                self.momentum * mean.reshape(-1)
            ).astype(np.float32)
            self.running_var = (
                (1.0 - self.momentum) * self.running_var +
                self.momentum * unbiased_var
            ).astype(np.float32)
        else:
            y_data = _core.batchnorm2d(x_tensor.data, self.gamma.data, self.beta.data, self.running_mean, self.running_var, self.eps)
        req_grad = x_tensor.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        out = Tensor(y_data, requires_grad=req_grad, device=x_tensor.device, _children=(x_tensor, self.gamma, self.beta))

        if req_grad:
            if self.training:
                inv_std = 1.0 / np.sqrt(var + self.eps)
                sample_count = x_tensor.data.shape[0] * x_tensor.data.shape[2] * x_tensor.data.shape[3]
            else:
                inv_std = 1.0 / np.sqrt(self.running_var.reshape(1, -1, 1, 1) + self.eps)
                x_hat = (x_tensor.data - self.running_mean.reshape(1, -1, 1, 1)) * inv_std

            def _backward():
                if x_tensor.requires_grad:
                    grad_xhat = out.grad * self.gamma.data.reshape(1, -1, 1, 1)
                    if self.training:
                        gx = (sample_count * grad_xhat -
                              np.sum(grad_xhat, axis=(0, 2, 3), keepdims=True) -
                              x_hat * np.sum(grad_xhat * x_hat, axis=(0, 2, 3), keepdims=True))
                        gx = gx * inv_std / sample_count
                    else:
                        gx = grad_xhat * inv_std
                    x_tensor._add_grad(gx)
                if self.gamma.requires_grad:
                    self.gamma._add_grad(np.sum(out.grad * x_hat, axis=(0, 2, 3)))
                if self.beta.requires_grad:
                    self.beta._add_grad(np.sum(out.grad, axis=(0, 2, 3)))

            out._backward = _backward

        return out


class GELU(Module):
    """Gaussian Error Linear Unit (GELU) aktivasyon fonksiyonu."""
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        y_data = _core.gelu(x_tensor.data)
        out = Tensor(y_data, requires_grad=x_tensor.requires_grad, device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                # Ileri gecis (C++ gelu_fn) tanh yaklasimini kullanir; turev de AYNI fonksiyonun turevi olmali.
                xd = x_tensor.data
                c = np.float32(np.sqrt(2.0 / np.pi))
                t = np.tanh(c * (xd + 0.044715 * xd ** 3))
                d_gelu = 0.5 * (1.0 + t) + 0.5 * xd * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044715 * xd ** 2)
                gx = out.grad * d_gelu
                x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class LayerNorm(Module):
    """Katman Normalizasyonu (Layer Normalization)."""
    def __init__(self, normalized_shape: int, eps: float = 1e-5, device: str = "cpu"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.device = device.lower()
        self.gamma = Parameter(np.ones(normalized_shape, dtype=np.float32), device=self.device)
        self.beta = Parameter(np.zeros(normalized_shape, dtype=np.float32), device=self.device)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        y_data = _core.layernorm(x_tensor.data, self.gamma.data, self.beta.data, self.eps)
        req_grad = x_tensor.requires_grad or self.gamma.requires_grad or self.beta.requires_grad
        out = Tensor(y_data, requires_grad=req_grad, device=x_tensor.device, _children=(x_tensor, self.gamma, self.beta))

        if req_grad:
            mean = np.mean(x_tensor.data, axis=-1, keepdims=True)
            var = np.var(x_tensor.data, axis=-1, keepdims=True)
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = (x_tensor.data - mean) * inv_std
            feature_dim = x_tensor.data.shape[-1]

            def _backward():
                grad = out.grad
                reduce_axes = tuple(range(grad.ndim - 1))
                if self.gamma.requires_grad:
                    self.gamma._add_grad(np.sum(grad * x_hat, axis=reduce_axes))
                if self.beta.requires_grad:
                    self.beta._add_grad(np.sum(grad, axis=reduce_axes))
                if x_tensor.requires_grad:
                    grad_xhat = grad * self.gamma.data
                    gx = (feature_dim * grad_xhat - np.sum(grad_xhat, axis=-1, keepdims=True) -
                          x_hat * np.sum(grad_xhat * x_hat, axis=-1, keepdims=True))
                    gx = gx * inv_std / feature_dim
                    x_tensor._add_grad(gx)

            out._backward = _backward

        return out


class MultiHeadAttention(Module):
    """Çok Başlı Öz-Dikkat (Multi-Head Self-Attention - MHSA) Katmanı."""
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, device: str = "cpu"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) num_heads ({num_heads}) değerine tam bölünmeli")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device.lower()

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, device=self.device)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, device=self.device)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, device=self.device)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=self.device)

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        shape = x_tensor.data.shape
        if len(shape) == 2:
            N, S, D = 1, shape[0], shape[1]
        else:
            N, S, D = shape[0], shape[1], shape[2]

        q = self.q_proj(x_tensor)
        k = self.k_proj(x_tensor)
        v = self.v_proj(x_tensor)

        q_h = q.reshape(N, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_h = k.reshape(N, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_h = v.reshape(N, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / np.sqrt(self.head_dim, dtype=np.float32)
        scores = (q_h @ k_h.transpose(0, 1, 3, 2)) * scale
        attn_weights = scores.softmax(axis=-1)
        attn_tensor = (attn_weights @ v_h).transpose(0, 2, 1, 3).reshape(shape)
        return self.out_proj(attn_tensor)


class Flatten(Module):
    """start_dim..end_dim eksenlerini tek eksende birleştirir (torch.nn.Flatten ile aynı).

    `Tensor.reshape` kullandığı için autograd grafı korunur; `nv.Tensor(x.data.reshape(...))`
    gibi elle düzleştirme ise grafı koparır ve önceki katmanlar eğitilmez.
    """
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        shape = x_tensor.data.shape
        ndim = len(shape)
        start = self.start_dim % ndim
        end = self.end_dim % ndim
        if start > end:
            raise ValueError(f"Flatten: start_dim ({self.start_dim}) end_dim'den ({self.end_dim}) büyük olamaz")
        new_shape = shape[:start] + (int(np.prod(shape[start:end + 1])),) + shape[end + 1:]
        return x_tensor.reshape(new_shape)


class Embedding(Module):
    """Tam sayı indekslerini öğrenilebilir vektörlere eşler (torch.nn.Embedding ile aynı).

    Ağırlık N(0, 1) ile başlatılır. padding_idx verilirse o satır sıfırla başlar ve gradyan almaz.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, device: str = "cpu"):
        super().__init__()
        if padding_idx is not None:
            if not -num_embeddings <= padding_idx < num_embeddings:
                raise ValueError(f"padding_idx [{-num_embeddings}, {num_embeddings}) aralığında olmalı, alınan: {padding_idx}")
            padding_idx = padding_idx % num_embeddings
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.device = device.lower()
        weight = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        if padding_idx is not None:
            weight[padding_idx] = 0.0
        self.weight = Parameter(weight, device=self.device)

    def forward(self, indices: Tensor | np.ndarray) -> Tensor:
        raw = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
        idx = raw.astype(np.int64)
        # Tensor verisi float32 tutulduğu için indeksin tam sayı olduğunu doğrula
        if not np.array_equal(idx, raw):
            raise TypeError("Embedding indeksleri tam sayı olmalıdır")
        if idx.size and (idx.min() < 0 or idx.max() >= self.num_embeddings):
            raise IndexError(f"Embedding indeksi [0, {self.num_embeddings}) aralığı dışında: min={idx.min()}, max={idx.max()}")

        out = Tensor(self.weight.data[idx], requires_grad=self.weight.requires_grad, device=self.device, _children=(self.weight,))

        if self.weight.requires_grad:
            def _backward():
                g = np.zeros_like(self.weight.data)
                np.add.at(g, idx, out.grad)
                if self.padding_idx is not None:
                    g[self.padding_idx] = 0.0
                self.weight._add_grad(g)

            out._backward = _backward

        return out


class BatchNorm1d(BatchNorm2d):
    """(N, C) veya (N, C, L) girdiler için batch normalizasyonu (torch.nn.BatchNorm1d ile aynı).

    Girdiyi (N, C, L, 1) şekline getirip BatchNorm2d'nin forward/backward yolunu kullanır.
    """
    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        shape = x_tensor.data.shape
        if len(shape) == 2:
            x4 = x_tensor.reshape(shape[0], shape[1], 1, 1)
        elif len(shape) == 3:
            x4 = x_tensor.reshape(shape[0], shape[1], shape[2], 1)
        else:
            raise ValueError(f"BatchNorm1d (N, C) veya (N, C, L) girdi bekler, alınan: {shape}")
        return super().forward(x4).reshape(shape)


class Conv1d(Module):
    """1D evrişim (torch.nn.Conv1d ile aynı ağırlık düzeni: (out, in // groups, k)).

    C++ conv2d çekirdeğini H=1 ile kullanır. C++ dolgusu iki eksene birden uygulandığı için
    dolgu yalnızca uzunluk ekseninde NumPy ile yapılır.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True, device: str = "cpu"):
        super().__init__()
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) ve out_channels ({out_channels}) groups'a ({groups}) bölünebilmeli")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.device = device.lower()

        scale = np.sqrt(2.0 / (in_channels * kernel_size / groups), dtype=np.float32)
        self.w = Parameter(np.random.randn(out_channels, in_channels // groups, kernel_size).astype(np.float32) * scale, device=self.device)
        if bias:
            self.bias = Parameter(np.zeros(out_channels, dtype=np.float32), device=self.device)
        else:
            self.bias = None

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x, device=self.device)
        if x_tensor.data.ndim != 3:
            raise ValueError(f"[Conv1d] Girdi (N, C, L) olmalı, alınan: {x_tensor.data.shape}")
        if x_tensor.data.shape[1] != self.in_channels:
            raise ValueError(f"[Conv1d] Girdi kanal sayısı ({x_tensor.data.shape[1]}) in_channels ({self.in_channels}) ile eşleşmiyor")

        p = self.padding
        xp = np.pad(x_tensor.data, ((0, 0), (0, 0), (p, p))) if p > 0 else x_tensor.data
        x4 = np.ascontiguousarray(xp[:, :, None, :])
        w4 = np.ascontiguousarray(self.w.data[:, :, None, :])
        b_data = self.bias.data if self.bias is not None else None
        y4 = _core.conv2d(x4, w4, b_data, self.stride, 0, self.dilation, self.groups, False)

        req_grad = x_tensor.requires_grad or self.w.requires_grad or (self.bias is not None and self.bias.requires_grad)
        children = [x_tensor, self.w] + ([self.bias] if self.bias is not None else [])
        out = Tensor(y4[:, :, 0, :], requires_grad=req_grad, device=x_tensor.device, _children=children)

        if req_grad:
            def _backward():
                g4 = np.ascontiguousarray(out.grad[:, :, None, :])
                gx4, gw4, gb = _core.conv2d_backward(g4, x4, w4, self.bias is not None, self.stride, 0, self.dilation, self.groups)
                if x_tensor.requires_grad:
                    gx = gx4[:, :, 0, :]
                    x_tensor._add_grad(gx[:, :, p:-p] if p > 0 else gx)
                if self.w.requires_grad:
                    self.w._add_grad(gw4[:, :, 0, :])
                if self.bias is not None and self.bias.requires_grad and gb is not None:
                    self.bias._add_grad(gb)

            out._backward = _backward

        return out


def _linear_interp_matrix(in_size: int, scale: int, align_corners: bool) -> np.ndarray:
    """(out, in) doğrusal interpolasyon matrisi; PyTorch'un koordinat eşlemesiyle aynı."""
    out_size = in_size * scale
    dst = np.arange(out_size, dtype=np.float64)
    if align_corners:
        src = dst * (in_size - 1) / (out_size - 1) if out_size > 1 else np.zeros_like(dst)
    else:
        src = np.maximum((dst + 0.5) / scale - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    frac = src - i0
    m = np.zeros((out_size, in_size), dtype=np.float64)
    np.add.at(m, (np.arange(out_size), i0), 1.0 - frac)
    np.add.at(m, (np.arange(out_size), i1), frac)
    return m.astype(np.float32)


class Upsample(Module):
    """Büyütme katmanı (torch.nn.Upsample ile aynı semantik), tam sayı scale_factor.

    mode: "nearest" ((N, C, L) veya (N, C, H, W)), "linear" ((N, C, L)), "bilinear" ((N, C, H, W)).
    align_corners yalnızca linear/bilinear için geçerlidir (varsayılan False, PyTorch gibi).
    """
    _MODES = ("nearest", "linear", "bilinear")

    def __init__(self, scale_factor: int, mode: str = "nearest", align_corners: bool | None = None):
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"Upsample mode {self._MODES} değerlerinden biri olmalı, alınan: {mode!r}")
        if not isinstance(scale_factor, (int, np.integer)) or scale_factor < 1:
            raise ValueError(f"Upsample scale_factor pozitif tam sayı olmalı, alınan: {scale_factor!r}")
        if mode == "nearest" and align_corners is not None:
            raise ValueError("align_corners yalnızca 'linear' ve 'bilinear' modlarında kullanılabilir")
        self.scale_factor = int(scale_factor)
        self.mode = mode
        self.align_corners = bool(align_corners) if align_corners is not None else False

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        x_tensor = x if isinstance(x, Tensor) else Tensor(x)
        s = self.scale_factor
        shape = x_tensor.data.shape
        expected = {"nearest": (3, 4), "linear": (3,), "bilinear": (4,)}[self.mode]
        if len(shape) not in expected:
            raise ValueError(f"Upsample mode={self.mode!r} {[f'{d}D' for d in expected]} girdi bekler, alınan: {shape}")

        if self.mode == "nearest":
            y = x_tensor.data
            for axis in range(2, len(shape)):
                y = np.repeat(y, s, axis=axis)
        else:
            # Ayrılabilir interpolasyon: her uzamsal eksen için (out, in) matrisiyle çarp
            mats = [_linear_interp_matrix(dim, s, self.align_corners) for dim in shape[2:]]
            y = x_tensor.data @ mats[-1].T                       # son eksen (W veya L)
            if len(mats) == 2:
                y = np.einsum("oh,nchw->ncow", mats[0], y)       # H ekseni
        out = Tensor(np.ascontiguousarray(y, dtype=np.float32), requires_grad=x_tensor.requires_grad,
                     device=x_tensor.device, _children=(x_tensor,))

        if x_tensor.requires_grad:
            def _backward():
                g = out.grad
                if self.mode == "nearest":
                    # Her çıktı bloğu tek bir girdi elemanından kopyalandı: blok gradyanlarını topla
                    blocked = g.reshape(shape[:2] + tuple(d for dim in shape[2:] for d in (dim, s)))
                    x_tensor._add_grad(blocked.sum(axis=tuple(range(3, blocked.ndim, 2))))
                else:
                    if len(mats) == 2:
                        g = np.einsum("oh,ncow->nchw", mats[0], g)
                    x_tensor._add_grad(g @ mats[-1])

            out._backward = _backward

        return out
