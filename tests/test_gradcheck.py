"""newton-vision Sayısal Gradyan Doğrulama (gradcheck) Testleri.

Sonlu farklar (finite differences) yöntemi ile analitik Autograd gradyanlarını
matematiksel olarak karşılaştırır ve doğrular:
    grad_numerical ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
"""

import unittest
import numpy as np
import newton_vision as nv


def numerical_gradient(fn, x_data: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Tek bir tensör girdisi alan skaler çıkışlı fonksiyonun sayısal gradyanını hesaplar."""
    grad = np.zeros_like(x_data, dtype=np.float64)
    it = np.nditer(x_data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = x_data[idx]

        x_data[idx] = orig_val + eps
        pos_out = float(fn(x_data).data)

        x_data[idx] = orig_val - eps
        neg_out = float(fn(x_data).data)

        x_data[idx] = orig_val
        grad[idx] = (pos_out - neg_out) / (2.0 * eps)
        it.iternext()
    return grad.astype(np.float32)


class TestGradCheck(unittest.TestCase):

    def test_linear_gradcheck(self):
        """Linear katmanı için x, w ve bias sayısal gradyan doğrulaması."""
        np.random.seed(42)
        in_feat, out_feat = 3, 2
        linear = nv.Linear(in_feat, out_feat, bias=True)

        x_val = np.random.randn(2, in_feat).astype(np.float32)
        x = nv.Tensor(x_val, requires_grad=True)

        # Skaler kayıp: out.sum()
        out = linear(x)
        loss = out.sum()
        loss.backward()

        # x için gradcheck
        def fn_x(curr_x):
            return (linear(nv.Tensor(curr_x))).sum()

        num_grad_x = numerical_gradient(fn_x, x_val.copy())
        diff_x = np.max(np.abs(x.grad - num_grad_x))
        self.assertLess(diff_x, 2e-3, f"Linear x gradyan farkı çok yüksek: {diff_x}")

        # w için gradcheck
        w_val = linear.w.data.copy()
        def fn_w(curr_w):
            linear.w.data = curr_w.copy()
            res = (linear(nv.Tensor(x_val))).sum()
            linear.w.data = w_val.copy()
            return res

        num_grad_w = numerical_gradient(fn_w, w_val.copy())
        diff_w = np.max(np.abs(linear.w.grad - num_grad_w))
        self.assertLess(diff_w, 2e-3, f"Linear w gradyan farkı çok yüksek: {diff_w}")

    def test_conv2d_gradcheck(self):
        """Conv2d katmanı için x ve w sayısal gradyan doğrulaması."""
        conv = nv.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, bias=False)
        x_val = np.random.randn(1, 2, 4, 4).astype(np.float32)
        x = nv.Tensor(x_val, requires_grad=True)

        out = conv(x)
        loss = out.sum()
        loss.backward()

        # x için gradcheck
        def fn_x(curr_x):
            return (conv(nv.Tensor(curr_x))).sum()

        num_grad_x = numerical_gradient(fn_x, x_val.copy(), eps=1e-3)
        diff_x = np.max(np.abs(x.grad - num_grad_x))
        self.assertLess(diff_x, 2e-3, f"Conv2d x gradyan farkı çok yüksek: {diff_x}")

    def test_layernorm_gradcheck(self):
        """LayerNorm katmanı için analitik ve sayısal gradyan doğrulaması."""
        np.random.seed(42)
        ln = nv.LayerNorm(normalized_shape=4)
        x_val = np.random.randn(2, 3, 4).astype(np.float32)
        x = nv.Tensor(x_val, requires_grad=True)

        # Ağırlıklı skaler toplam ile test ediyoruz
        weights = np.random.randn(2, 3, 4).astype(np.float32)
        out = ln(x)
        loss = (out * weights).sum()
        loss.backward()

        def fn_x(curr_x):
            return (ln(nv.Tensor(curr_x)) * weights).sum()

        num_grad_x = numerical_gradient(fn_x, x_val.copy(), eps=1e-4)
        diff_x = np.max(np.abs(x.grad - num_grad_x))
        self.assertLess(diff_x, 6e-3, f"LayerNorm x gradyan farkı çok yüksek: {diff_x}")

    def test_gelu_gradcheck(self):
        """GELU aktivasyon fonksiyonu için sayısal gradyan doğrulaması."""
        gelu = nv.optics.GELU()
        x_val = np.array([-1.5, -0.5, 0.2, 1.0, 2.5], dtype=np.float32)
        x = nv.Tensor(x_val, requires_grad=True)

        out = gelu(x)
        loss = out.sum()
        loss.backward()

        def fn_x(curr_x):
            return gelu(nv.Tensor(curr_x)).sum()

        num_grad_x = numerical_gradient(fn_x, x_val.copy(), eps=1e-4)
        diff_x = np.max(np.abs(x.grad - num_grad_x))
        self.assertLess(diff_x, 1e-3, f"GELU gradyan farkı çok yüksek: {diff_x}")


if __name__ == "__main__":
    unittest.main()
