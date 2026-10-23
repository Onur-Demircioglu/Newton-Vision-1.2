from __future__ import annotations
import numpy as np


_GRAD_ENABLED = True


def is_grad_enabled() -> bool:
    """Global Autograd türev takibinin aktif olup olmadığını döndürür."""
    global _GRAD_ENABLED
    return _GRAD_ENABLED


def set_grad_enabled(mode: bool):
    """Global Autograd türev takibi modunu ayarlar."""
    global _GRAD_ENABLED
    _GRAD_ENABLED = mode


class no_grad:
    """Gradyan hesaplamasını devre dışı bırakan context manager ve decorator."""
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            with no_grad():
                return fn(*args, **kwargs)
        return wrapper


class enable_grad:
    """Gradyan hesaplamasını zorla aktif kılan context manager ve decorator."""
    def __enter__(self):
        self.prev = is_grad_enabled()
        set_grad_enabled(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_grad_enabled(self.prev)

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            with enable_grad():
                return fn(*args, **kwargs)
        return wrapper


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad: bool = False, device: str = "cpu", _children=()):
        self.data = np.asarray(data, dtype=np.float32)
        # Eğer global gradyan kapalıysa (no_grad aktifse), operasyonlar türev grafına bağlanmaz
        effective_requires_grad = requires_grad and is_grad_enabled()
        self.grad = np.zeros_like(self.data, dtype=np.float32) if effective_requires_grad else None
        self.requires_grad = effective_requires_grad
        self.device = device.lower()
        self._prev = set(_children) if effective_requires_grad else set()
        self._backward = lambda: None

    def _add_grad(self, grad: np.ndarray):
        grad = np.asarray(grad, dtype=np.float32)
        if grad.shape != self.data.shape:
            grad = _unbroadcast(grad, self.data.shape)
        self.grad = self.grad + grad if self.grad is not None else grad.copy()

    def _ensure_compatible(self, other: Tensor):
        if self.device != other.device:
            raise RuntimeError(f"Tensor cihazlari uyusmuyor: {self.device} vs {other.device}")

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def strides(self):
        return self.data.strides

    def item(self):
        """Tek elemanlı tensörü skaler Python float/int değerine dönüştürür."""
        if self.data.size != 1:
            raise ValueError(f"Sadece tek elemanlı tensörler skaler değere dönüştürülebilir, mevcut boyut: {self.data.size}")
        return self.data.item()

    def numpy(self) -> np.ndarray:
        """Tensörün NumPy ndarray verisini döndürür."""
        return self.data

    def __len__(self) -> int:
        if self.data.ndim == 0:
            raise TypeError("len() 0-boyutlu tensör için çağrılamaz")
        return self.data.shape[0]

    def to(self, device: str) -> Tensor:
        """Tensörü istenen cihaza geçiren metod ('cpu' veya 'cuda')."""
        dev = device.lower()
        if dev not in ("cpu", "cuda"):
            raise ValueError(f"Bilinmeyen cihaz: {device}. Beklenen 'cpu' veya 'cuda'")
        if dev == "cuda":
            try:
                import newton_vision._core as _core
                has_cuda = _core.has_cuda()
            except Exception:
                has_cuda = False
            if not has_cuda:
                raise RuntimeError("CUDA bu build/cihaz icin kullanilabilir degil; Tensor CPU'da kalmali.")
        if dev == self.device:
            return self

        # Cihaz geçişi türevlenebilir bir kimlik işlemidir: gradyan kaynak tensöre aynen akar.
        # (Eski hali kaynak düğümün _backward closure'ını kopyalıyordu; gradyan yeni düğüme
        # değil eski düğümün çıktısına yazılıyor ve zincir kopuyordu.)
        out = Tensor(self.data, requires_grad=self.requires_grad, device=dev, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad)

            out._backward = _backward

        return out

    def cuda(self) -> Tensor:
        """Tensörü CUDA GPU VRAM belleğine aktarır."""
        return self.to("cuda")

    def cpu(self) -> Tensor:
        """Tensörü RAM CPU belleğine aktarır."""
        return self.to("cpu")

    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0.0)

    def backward(self, grad=None):
        if not self.requires_grad:
            return

        if grad is None:
            if self.data.size == 1:
                grad = np.ones_like(self.data, dtype=np.float32)
            else:
                raise RuntimeError("grad non-scalar çıktılar için açıkça belirtilmelidir")
        else:
            grad = np.asarray(grad, dtype=np.float32)

        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

        # İteratif (yığın tabanlı) topolojik sıralama: özyineleme ~1000 düğümden
        # uzun graflarda RecursionError veriyordu.
        topo = []
        visited = {id(self)}
        stack = [(self, iter(self._prev))]
        while stack:
            node, children = stack[-1]
            for child in children:
                if id(child) not in visited:
                    visited.add(id(child))
                    stack.append((child, iter(child._prev)))
                    break
            else:
                stack.pop()
                topo.append(node)

        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._ensure_compatible(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad), device=self.device, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad)
            if other.requires_grad:
                other._add_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._ensure_compatible(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad), device=self.device, _children=(self, other))

        def _backward():
            if self.requires_grad:
                g = out.grad * other.data
                self._add_grad(g)
            if other.requires_grad:
                g = out.grad * self.data
                other._add_grad(g)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        self._ensure_compatible(other)
        out = Tensor(self.data @ other.data, requires_grad=(self.requires_grad or other.requires_grad), device=self.device, _children=(self, other))

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad @ np.swapaxes(other.data, -1, -2))
            if other.requires_grad:
                other._add_grad(np.swapaxes(self.data, -1, -2) @ out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return self + (other * -1.0)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other - self

    def __neg__(self):
        return self * -1.0

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other / self

    def __pow__(self, power: float):
        out = Tensor(self.data ** power, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad * power * (self.data ** (power - 1.0)))

            out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                g = np.ones_like(self.data, dtype=np.float32) * grad
                self._add_grad(g)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        def _backward():
            if self.requires_grad:
                if axis is None:
                    denom = self.data.size
                else:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                    denom = int(np.prod([self.data.shape[a] for a in axes]))
                grad = out.grad
                if axis is not None and not keepdims:
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)
                g = (np.ones_like(self.data, dtype=np.float32) / denom) * grad
                self._add_grad(g)

        out._backward = _backward
        return out

    def __getitem__(self, idx):
        """Sıfır-Kopya Strided View Dilimleme."""
        view_data = self.data[idx]
        out = Tensor(view_data, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                grad = np.zeros_like(self.data, dtype=np.float32)
                np.add.at(grad, idx, out.grad)
                self._add_grad(grad)

            out._backward = _backward

        return out

    def reshape(self, *shape) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad.reshape(self.data.shape))

            out._backward = _backward

        return out

    def transpose(self, *axes) -> Tensor:
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        if len(axes) == 0:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            inverse_axes = np.argsort(axes)

            def _backward():
                self._add_grad(out.grad.transpose(*inverse_axes))

            out._backward = _backward

        return out

    def max(self, axis=None, keepdims: bool = False) -> Tensor:
        max_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(max_data, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                grad = out.grad
                max_values = max_data
                axes = None
                if axis is not None:
                    axes = axis if isinstance(axis, tuple) else (axis,)
                    axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                    if not keepdims:
                        for ax in sorted(axes):
                            grad = np.expand_dims(grad, ax)
                            max_values = np.expand_dims(max_values, ax)
                mask = self.data == max_values
                count = np.sum(mask, axis=axes, keepdims=True) if axes is not None else np.sum(mask)
                self._add_grad(mask.astype(np.float32) * grad / count)

            out._backward = _backward

        return out

    def exp(self) -> Tensor:
        y = np.exp(self.data)
        out = Tensor(y, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad * y)

            out._backward = _backward

        return out

    def log(self) -> Tensor:
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad / self.data)

            out._backward = _backward

        return out

    def sqrt(self) -> Tensor:
        y = np.sqrt(self.data)
        out = Tensor(y, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad * 0.5 / y)

            out._backward = _backward

        return out

    def tanh(self) -> Tensor:
        y = np.tanh(self.data)
        out = Tensor(y, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                self._add_grad(out.grad * (1.0 - y ** 2))

            out._backward = _backward

        return out

    def softmax(self, axis: int = -1) -> Tensor:
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp_data = np.exp(shifted)
        y = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        out = Tensor(y, requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            def _backward():
                dot = np.sum(out.grad * y, axis=axis, keepdims=True)
                self._add_grad(y * (out.grad - dot))

            out._backward = _backward

        return out

    def tile(self, reps) -> Tensor:
        out = Tensor(np.tile(self.data, reps), requires_grad=self.requires_grad, device=self.device, _children=(self,))

        if self.requires_grad:
            reps_tuple = reps if isinstance(reps, tuple) else (reps,)
            padded_shape = (1,) * max(0, len(reps_tuple) - self.data.ndim) + self.data.shape
            padded_reps = (1,) * max(0, len(padded_shape) - len(reps_tuple)) + reps_tuple

            def _backward():
                shape = tuple(dim for pair in zip(padded_reps, padded_shape) for dim in pair)
                grad = out.grad.reshape(shape)
                for axis_idx in range(len(padded_reps) - 1, -1, -1):
                    grad = grad.sum(axis=axis_idx * 2)
                self._add_grad(grad.reshape(self.data.shape))

            out._backward = _backward

        return out

    @staticmethod
    def concatenate(tensors, axis: int = 0) -> Tensor:
        tensor_list = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        if not tensor_list:
            raise ValueError("concatenate en az bir Tensor bekler")
        device = tensor_list[0].device
        for tensor in tensor_list[1:]:
            if tensor.device != device:
                raise RuntimeError(f"Tensor cihazlari uyusmuyor: {device} vs {tensor.device}")

        out = Tensor(
            np.concatenate([t.data for t in tensor_list], axis=axis),
            requires_grad=any(t.requires_grad for t in tensor_list),
            device=device,
            _children=tuple(tensor_list),
        )

        def _backward():
            start = 0
            for tensor in tensor_list:
                size = tensor.data.shape[axis]
                if tensor.requires_grad:
                    index = [slice(None)] * out.grad.ndim
                    index[axis] = slice(start, start + size)
                    tensor._add_grad(out.grad[tuple(index)])
                start += size

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor({self.data}, device='{self.device}', requires_grad={self.requires_grad})"
