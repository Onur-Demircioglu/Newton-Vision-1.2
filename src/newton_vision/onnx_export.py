"""newton_vision.onnx_export - İzleme (tracing) tabanlı gerçek ONNX dışa aktarımı.

Model örnek bir girdiyle bir kez çalıştırılır; bu sırada:
  * Dönüştürücüsü olan newton-vision katmanları (Conv2d, Linear, BatchNorm2d, ...) tek ONNX düğümü
    olarak kaydedilir (iç işlemleri izlenmez).
  * Kullanıcı / model zoo `forward` metotlarındaki Tensor işlemleri (+, *, @, reshape, transpose,
    softmax, concatenate, tile, dilimleme, ...) ONNX düğümlerine çevrilir.
  * Parameter'lar ve sabitler initializer olur.

Sınırlar:
  * Graf sabit şekillidir (örnek girdinin şekli, batch boyutu dahil).
  * Python kontrol akışı izleme anındaki dala göre sabitlenir.
  * Dönüştürücüsü olmayan newton-vision katmanları (ör. QuantizedLinear) NotImplementedError verir.
Doğruluk, onnxruntime kuruluysa iki farklı girdiyle newton-vision çıktısına karşı doğrulanır.
"""

from __future__ import annotations
import contextlib
import numpy as np

from newton_vision.tensor import Tensor
from newton_vision.module import Module, Parameter
from newton_vision import layers as L

OPSET = 17


class _Graph:
    def __init__(self, onnx_mod):
        self.onnx = onnx_mod
        self.helper = onnx_mod.helper
        self.nodes = []
        self.initializers = []
        self.names: dict[int, str] = {}
        self.keep_alive: list = []   # id() yeniden kullanılmasın diye izlenen nesneleri tut
        self.counter = 0
        self.leaf_depth = 0

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def const(self, value, dtype=np.float32, prefix: str = "const") -> str:
        arr = np.asarray(value, dtype=dtype)
        name = self.fresh(prefix)
        self.initializers.append(self.onnx.numpy_helper.from_array(arr, name))
        return name

    def name_of(self, t) -> str:
        """Tensor'ın ONNX değer adı; izlenmemişse (Parameter / sabit) initializer olarak eklenir."""
        if isinstance(t, Tensor):
            if id(t) in self.names:
                return self.names[id(t)]
            prefix = "param" if isinstance(t, Parameter) else "const"
            name = self.const(t.data, prefix=prefix)
            self.bind(t, name)
            return name
        return self.const(t)

    def bind(self, t: Tensor, name: str):
        self.names[id(t)] = name
        self.keep_alive.append(t)

    def node(self, op: str, inputs: list[str], out: Tensor | None = None, n_out: int = 1, **attrs) -> str:
        outputs = [self.fresh(op.lower()) for _ in range(n_out)]
        self.nodes.append(self.helper.make_node(op, inputs, outputs, **attrs))
        if out is not None:
            self.bind(out, outputs[0])
        return outputs[0]


# ----------------------------------------------------------------------------- katman dönüştürücüleri

def _conv2d(g, m: L.Conv2d, x, out):
    kH, kW = m.kernel_size
    p, s, d = m.padding, m.stride, m.dilation
    inputs = [g.name_of(x), g.name_of(m.w)] + ([g.name_of(m.bias)] if m.bias is not None else [])
    g.node("Conv", inputs, out, kernel_shape=[kH, kW], strides=[s, s], pads=[p, p, p, p], dilations=[d, d], group=m.groups)


def _conv1d(g, m: L.Conv1d, x, out):
    p, s, d = m.padding, m.stride, m.dilation
    inputs = [g.name_of(x), g.name_of(m.w)] + ([g.name_of(m.bias)] if m.bias is not None else [])
    g.node("Conv", inputs, out, kernel_shape=[m.kernel_size], strides=[s], pads=[p, p], dilations=[d], group=m.groups)


def _linear(g, m: L.Linear, x, out):
    # newton-vision ağırlığı (in, out): y = x @ W + b, her rank için MatMul ile
    y = g.node("MatMul", [g.name_of(x), g.name_of(m.w)])
    if m.bias is not None:
        g.node("Add", [y, g.name_of(m.bias)], out)
    else:
        g.node("Identity", [y], out)


def _gelu(g, m, x, out):
    # GELU (tanh yaklaşımı, C++ çekirdeğiyle aynı): 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
    xn = g.name_of(x)
    x3 = g.node("Mul", [g.node("Mul", [xn, xn]), xn])
    inner = g.node("Add", [xn, g.node("Mul", [x3, g.const(0.044715)])])
    th = g.node("Tanh", [g.node("Mul", [inner, g.const(0.7978845608028654)])])
    half_x = g.node("Mul", [xn, g.const(0.5)])
    g.node("Mul", [half_x, g.node("Add", [th, g.const(1.0)])], out)


def _batchnorm(g, m: L.BatchNorm2d, x, out):
    if m.training:
        raise RuntimeError("ONNX dışa aktarımı eval modunda yapılmalı (BatchNorm running istatistikleri)")
    g.node("BatchNormalization", [g.name_of(x), g.name_of(m.gamma), g.name_of(m.beta),
                                  g.const(m.running_mean), g.const(m.running_var)], out, epsilon=float(m.eps))


def _layernorm(g, m: L.LayerNorm, x, out):
    g.node("LayerNormalization", [g.name_of(x), g.name_of(m.gamma), g.name_of(m.beta)], out, axis=-1, epsilon=float(m.eps))


def _maxpool(g, m: L.MaxPool2d, x, out):
    k, s, p = m.kernel_size, m.stride, m.padding
    g.node("MaxPool", [g.name_of(x)], out, kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p])


def _avgpool(g, m: L.AvgPool2d, x, out):
    k, s = m.kernel_size, m.stride
    g.node("AveragePool", [g.name_of(x)], out, kernel_shape=[k, k], strides=[s, s])


def _upsample(g, m: L.Upsample, x, out):
    nd = len(x.shape)
    scales = g.const([1.0, 1.0] + [float(m.scale_factor)] * (nd - 2))
    roi = g.const([], prefix="roi")
    if m.mode == "nearest":
        g.node("Resize", [g.name_of(x), roi, scales], out, mode="nearest",
               coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    else:
        g.node("Resize", [g.name_of(x), roi, scales], out, mode="linear",
               coordinate_transformation_mode="align_corners" if m.align_corners else "pytorch_half_pixel")


def _embedding(g, m: L.Embedding, x, out):
    idx = g.name_of(x)
    if not getattr(g, "int_inputs", set()) or idx not in g.int_inputs:
        idx = g.node("Cast", [idx], to=g.onnx.TensorProto.INT64)
    g.node("Gather", [g.name_of(m.weight), idx], out, axis=0)


def _unary(op, **attrs):
    def conv(g, m, x, out):
        g.node(op, [g.name_of(x)], out, **attrs)
    return conv


def _flatten(g, m: L.Flatten, x, out):
    g.node("Reshape", [g.name_of(x), g.const(out.shape, dtype=np.int64)], out)


def _dropout(g, m: L.Dropout, x, out):
    g.node("Identity", [g.name_of(x)], out)


_CONVERTERS = {
    L.Conv2d: _conv2d, L.Conv1d: _conv1d, L.Linear: _linear,
    L.ReLU: _unary("Relu"), L.Sigmoid: _unary("Sigmoid"), L.Tanh: _unary("Tanh"), L.GELU: _gelu,
    L.BatchNorm2d: _batchnorm, L.BatchNorm1d: _batchnorm, L.LayerNorm: _layernorm,
    L.MaxPool2d: _maxpool, L.AvgPool2d: _avgpool, L.Upsample: _upsample, L.Embedding: _embedding,
    L.Flatten: _flatten, L.Dropout: _dropout,
}


def _leaky(g, m: L.LeakyReLU, x, out):
    g.node("LeakyRelu", [g.name_of(x)], out, alpha=float(m.negative_slope))


_CONVERTERS[L.LeakyReLU] = _leaky
_ATOMIC_MODULE_PREFIXES = ("newton_vision.layers", "newton_vision.quantization", "newton_vision.gravity")
_TRACE_THROUGH = {L.MultiHeadAttention}


# ----------------------------------------------------------------------------- Tensor işlem izleme

def _norm_axes(axis, ndim):
    if axis is None:
        return list(range(ndim))
    axes = axis if isinstance(axis, (tuple, list)) else (axis,)
    return [a % ndim for a in axes]


def _record_getitem(g, x: Tensor, idx, out: Tensor):
    idx = idx if isinstance(idx, tuple) else (idx,)
    if any(i is Ellipsis or i is None or not isinstance(i, (int, np.integer, slice)) for i in idx):
        raise NotImplementedError(f"ONNX: yalnızca int/slice ile temel dilimleme destekleniyor, alınan: {idx}")
    starts, ends, axes, steps, squeeze = [], [], [], [], []
    for ax, i in enumerate(idx):
        dim = x.shape[ax]
        if isinstance(i, slice):
            st, en, sp = i.indices(dim)
            if sp < 0:
                raise NotImplementedError("ONNX: negatif adımlı dilimleme desteklenmiyor")
            if (st, en, sp) == (0, dim, 1):
                continue
        else:
            st = int(i) % dim
            en, sp = st + 1, 1
            squeeze.append(ax)
        starts.append(st); ends.append(en); axes.append(ax); steps.append(sp)
    name = g.name_of(x)
    if axes:
        name = g.node("Slice", [name, g.const(starts, np.int64), g.const(ends, np.int64),
                                g.const(axes, np.int64), g.const(steps, np.int64)])
    if squeeze:
        name = g.node("Squeeze", [name, g.const(squeeze, np.int64)])
    g.node("Identity", [name], out)


def _make_patches(g):
    orig = {}

    def wrap(name, recorder):
        f = getattr(Tensor, name)
        orig[name] = f

        def patched(self, *args, **kwargs):
            if g.leaf_depth > 0:
                return f(self, *args, **kwargs)
            # Sayısal operandları kaydedilmiş sabit Tensor'a çevir (bileşik işlemler de izlensin)
            args = tuple(Tensor(a, device=self.device) if isinstance(a, (int, float, np.ndarray, np.number)) and name in _BINARY else a
                         for a in args)
            for a in args:
                if isinstance(a, Tensor) and id(a) not in g.names:
                    g.name_of(a)
            out = f(self, *args, **kwargs)
            if recorder is not None and id(out) not in g.names:
                recorder(self, args, kwargs, out)
            return out
        setattr(Tensor, name, patched)

    def binary(op, reverse=False):
        def rec(self, args, kwargs, out):
            a, b = (args[0], self) if reverse else (self, args[0])
            g.node(op, [g.name_of(a), g.name_of(b)], out)
        return rec

    wrap("__add__", binary("Add"))
    wrap("__mul__", binary("Mul"))
    wrap("__matmul__", binary("MatMul"))
    for composite in ("__sub__", "__rsub__", "__truediv__", "__rtruediv__", "__radd__", "__rmul__", "__neg__"):
        wrap(composite, None)   # içte izlenen temel işlemleri çağırır

    def rec_pow(self, args, kwargs, out):
        g.node("Pow", [g.name_of(self), g.const(float(args[0]))], out)
    wrap("__pow__", rec_pow)

    def rec_reshape(self, args, kwargs, out):
        g.node("Reshape", [g.name_of(self), g.const(out.shape, np.int64)], out)
    wrap("reshape", rec_reshape)

    def rec_transpose(self, args, kwargs, out):
        axes = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        perm = list(axes) if axes else list(reversed(range(self.ndim)))
        g.node("Transpose", [g.name_of(self)], out, perm=[int(a) for a in perm])
    wrap("transpose", rec_transpose)

    def reduce(op, axes_as_input):
        def rec(self, args, kwargs, out):
            axis = kwargs.get("axis", args[0] if args else None)
            keep = int(kwargs.get("keepdims", args[1] if len(args) > 1 else False))
            axes = _norm_axes(axis, self.ndim)
            if axes_as_input:
                g.node(op, [g.name_of(self), g.const(axes, np.int64)], out, keepdims=keep)
            else:
                g.node(op, [g.name_of(self)], out, axes=axes, keepdims=keep)
        return rec
    wrap("sum", reduce("ReduceSum", True))
    wrap("mean", reduce("ReduceMean", False))
    wrap("max", reduce("ReduceMax", False))

    for meth, op in (("exp", "Exp"), ("log", "Log"), ("sqrt", "Sqrt"), ("tanh", "Tanh")):
        wrap(meth, (lambda op_: lambda self, a, k, out: g.node(op_, [g.name_of(self)], out))(op))

    def rec_softmax(self, args, kwargs, out):
        axis = kwargs.get("axis", args[0] if args else -1)
        g.node("Softmax", [g.name_of(self)], out, axis=int(axis))
    wrap("softmax", rec_softmax)

    def rec_tile(self, args, kwargs, out):
        reps = args[0] if isinstance(args[0], tuple) else (args[0],)
        if len(reps) > self.ndim:
            raise NotImplementedError("ONNX: tile tekrar sayısı tensör boyutundan fazla olamaz")
        reps = (1,) * (self.ndim - len(reps)) + tuple(reps)
        g.node("Tile", [g.name_of(self), g.const(reps, np.int64)], out)
    wrap("tile", rec_tile)

    def rec_getitem(self, args, kwargs, out):
        _record_getitem(g, self, args[0], out)
    wrap("__getitem__", rec_getitem)

    def rec_to(self, args, kwargs, out):
        g.node("Identity", [g.name_of(self)], out)
    wrap("to", rec_to)

    orig_concat = Tensor.concatenate

    def concat(tensors, axis: int = 0):
        out = orig_concat(tensors, axis=axis)
        if g.leaf_depth == 0:
            ts = list(tensors)
            nd = out.ndim
            g.node("Concat", [g.name_of(t) for t in ts], out, axis=int(axis) % nd)
        return out
    Tensor.concatenate = staticmethod(concat)
    orig["concatenate"] = orig_concat

    orig_call = Module.__call__

    def module_call(self, *args, **kwargs):
        if g.leaf_depth > 0:
            return orig_call(self, *args, **kwargs)
        conv = _CONVERTERS.get(type(self))
        if conv is None:
            # Kendi forward'ı Tensor işlemleriyle yazılmış katmanlar içinden geçilerek izlenir
            if type(self) not in _TRACE_THROUGH and type(self).__module__.startswith(_ATOMIC_MODULE_PREFIXES):
                raise NotImplementedError(f"ONNX: '{type(self).__name__}' katmanı için dönüştürücü yok")
            return orig_call(self, *args, **kwargs)
        if len(args) != 1 or kwargs:
            raise NotImplementedError(f"ONNX: '{type(self).__name__}' tek girdili çağrılmalı")
        x = args[0] if isinstance(args[0], Tensor) else Tensor(args[0])
        g.leaf_depth += 1
        try:
            out = orig_call(self, x)
        finally:
            g.leaf_depth -= 1
        conv(g, self, x, out)
        return out

    Module.__call__ = module_call
    return orig, orig_call


_BINARY = {"__add__", "__mul__", "__matmul__", "__sub__", "__rsub__", "__truediv__", "__rtruediv__", "__radd__", "__rmul__"}


@contextlib.contextmanager
def _tracing(g):
    orig, orig_call = _make_patches(g)
    try:
        yield
    finally:
        for name, f in orig.items():
            setattr(Tensor, name, staticmethod(f) if name == "concatenate" else f)
        Module.__call__ = orig_call


def _set_train_mode(model, modes=None):
    if modes is None:
        modes = [(m, m.training) for m in model.modules()]
        model.eval()
        return modes
    for m, was in modes:
        m.training = was


def export(model: Module, dummy_input, filepath: str, verify: bool = True, opset: int = OPSET) -> str:
    try:
        import onnx
    except ImportError as e:
        raise ImportError("ONNX dışa aktarımı için 'onnx' paketi gerekli: pip install newton-vision[onnx]") from e
    from newton_vision.tensor import no_grad

    raw = dummy_input.data if isinstance(dummy_input, Tensor) else np.asarray(dummy_input)
    is_int = np.issubdtype(np.asarray(dummy_input).dtype, np.integer) if not isinstance(dummy_input, Tensor) else False
    g = _Graph(onnx)
    x = Tensor(raw)
    in_dtype = onnx.TensorProto.INT64 if is_int else onnx.TensorProto.FLOAT
    g.bind(x, "input")
    g.int_inputs = {"input"} if is_int else set()

    modes = _set_train_mode(model)
    try:
        with no_grad(), _tracing(g):
            y = model(x)
    finally:
        _set_train_mode(model, modes)
    if not isinstance(y, Tensor):
        raise NotImplementedError(f"ONNX: model tek bir Tensor döndürmeli, alınan: {type(y)}")
    if id(y) not in g.names:
        raise RuntimeError("ONNX: model çıktısı izlenen graftan üretilmedi")

    g.nodes.append(g.helper.make_node("Identity", [g.names[id(y)]], ["output"]))
    graph = g.helper.make_graph(
        g.nodes, "newton_vision_model",
        [g.helper.make_tensor_value_info("input", in_dtype, list(raw.shape))],
        [g.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(y.shape))],
        initializer=g.initializers,
    )
    proto = g.helper.make_model(graph, opset_imports=[g.helper.make_opsetid("", opset)], producer_name="newton-vision")
    proto.ir_version = 8   # opset 17 ile uyumlu, eski onnxruntime sürümleri de açabilir
    onnx.checker.check_model(proto)
    onnx.save(proto, filepath)

    if verify:
        _verify(model, filepath, raw, is_int)
    return filepath


def _verify(model, filepath, raw, is_int):
    try:
        import onnxruntime as ort
    except ImportError:
        import warnings
        warnings.warn("onnxruntime kurulu değil: dışa aktarılan model sayısal olarak doğrulanmadı", stacklevel=3)
        return
    from newton_vision.tensor import no_grad
    sess = ort.InferenceSession(filepath, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    if is_int:
        second = rng.integers(0, int(raw.max()) + 1, raw.shape)
    else:
        second = rng.standard_normal(raw.shape)
    modes = _set_train_mode(model)
    try:
        for sample in (raw, second):
            feed = np.asarray(sample, dtype=np.int64 if is_int else np.float32)
            with no_grad():
                expected = model(Tensor(feed)).data
            got = sess.run(None, {"input": feed})[0]
            if got.shape != expected.shape or not np.allclose(got, expected, atol=1e-4, rtol=1e-3):
                diff = float(np.abs(got - expected).max()) if got.shape == expected.shape else float("nan")
                raise RuntimeError(
                    f"ONNX doğrulaması başarısız: onnxruntime çıktısı newton-vision'dan farklı (max fark {diff}). "
                    "Model girdiye bağlı bir değeri Tensor(...) ile grafa sabit olarak gömüyor olabilir.")
    finally:
        _set_train_mode(model, modes)
