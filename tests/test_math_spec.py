"""docs/mathematics.md <-> kod baglayici spesifikasyon testleri.

Kurallar (gradcheck metodolojisi):
* Referans: PyTorch FP64. newton-vision FP32 calistigi icin tolerans FP32 seviyesindedir.
* Kayip L = sum(R * Y), R sabit rastgele. L = sum(Y) BN/LN/softmax icin ozdes sifir
  gradyan verir ve gamma hatalarini gizler.
* gamma != 1, beta != 0 kullanilir; aksi halde gamma'nin iki kez uygulanmasi gorunmez.
"""

import unittest
import numpy as np
import newton_vision as nv

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

RTOL = 1e-4
ATOL = 1e-5


def _t64(a, grad=False):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=grad)


@unittest.skipUnless(HAS_TORCH, "PyTorch referansi gerekli")
class TestMathSpec(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1234)

    # --- 8.1 BatchNorm2d (train) ---
    def test_batchnorm_train_backward_gamma_not_one(self):
        x = self.rng.standard_normal((4, 3, 5, 5)).astype(np.float32)
        gamma = self.rng.uniform(0.5, 2.0, 3).astype(np.float32)
        beta = self.rng.standard_normal(3).astype(np.float32)
        R = self.rng.standard_normal((4, 3, 5, 5)).astype(np.float32)

        bn = nv.BatchNorm2d(3)
        bn.gamma.data = gamma.copy()
        bn.beta.data = beta.copy()
        bn.train()
        xt = nv.Tensor(x, requires_grad=True)
        (bn(xt) * nv.Tensor(R)).sum().backward()

        tx, tg, tb = _t64(x, True), _t64(gamma, True), _t64(beta, True)
        ty = torch.nn.functional.batch_norm(tx, None, None, tg, tb, training=True, eps=1e-5)
        (ty * _t64(R)).sum().backward()

        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(bn.gamma.grad, tg.grad.numpy(), rtol=RTOL, atol=1e-4)
        np.testing.assert_allclose(bn.beta.grad, tb.grad.numpy(), rtol=RTOL, atol=1e-4)

    def test_batchnorm_running_var_unbiased_matches_torch(self):
        x = self.rng.standard_normal((3, 4, 2, 2)).astype(np.float32) * 2 + 1
        bn = nv.BatchNorm2d(4)
        bn.train()
        bn(nv.Tensor(x))
        tbn = torch.nn.BatchNorm2d(4).double().train()
        tbn(_t64(x))
        np.testing.assert_allclose(bn.running_mean, tbn.running_mean.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(bn.running_var, tbn.running_var.numpy(), rtol=1e-5, atol=1e-6)

    def test_batchnorm_single_value_per_channel_rejected(self):
        bn = nv.BatchNorm2d(2)
        bn.train()
        with self.assertRaises(ValueError):
            bn(nv.Tensor(np.ones((1, 2, 1, 1), dtype=np.float32)))

    def test_batchnorm_eval_backward(self):
        x = self.rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
        gamma = self.rng.uniform(0.5, 2.0, 3).astype(np.float32)
        rm = self.rng.standard_normal(3).astype(np.float32)
        rv = self.rng.uniform(0.5, 2.0, 3).astype(np.float32)
        R = self.rng.standard_normal(x.shape).astype(np.float32)
        bn = nv.BatchNorm2d(3)
        bn.gamma.data = gamma.copy()
        bn.running_mean, bn.running_var = rm.copy(), rv.copy()
        bn.eval()
        xt = nv.Tensor(x, requires_grad=True)
        (bn(xt) * nv.Tensor(R)).sum().backward()
        tx = _t64(x, True)
        ty = torch.nn.functional.batch_norm(tx, _t64(rm), _t64(rv), _t64(gamma), _t64(np.zeros(3)),
                                            training=False, eps=1e-5)
        (ty * _t64(R)).sum().backward()
        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- 8.2 LayerNorm (gamma eksen boyunca degisir, 3D giris) ---
    def test_layernorm_backward_gamma_varies_along_axis(self):
        x = self.rng.standard_normal((2, 7, 16)).astype(np.float32)
        gamma = self.rng.uniform(0.5, 2.0, 16).astype(np.float32)
        beta = self.rng.standard_normal(16).astype(np.float32)
        R = self.rng.standard_normal((2, 7, 16)).astype(np.float32)

        ln = nv.LayerNorm(16)
        ln.gamma.data = gamma.copy()
        ln.beta.data = beta.copy()
        xt = nv.Tensor(x, requires_grad=True)
        (ln(xt) * nv.Tensor(R)).sum().backward()

        tx, tg, tb = _t64(x, True), _t64(gamma, True), _t64(beta, True)
        ty = torch.nn.functional.layer_norm(tx, (16,), tg, tb, eps=1e-5)
        (ty * _t64(R)).sum().backward()

        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)
        # gamma/beta gradyani batch VE token eksenleri uzerinden toplanir -> sekil (D,)
        self.assertEqual(ln.gamma.grad.shape, (16,))
        np.testing.assert_allclose(ln.gamma.grad, tg.grad.numpy(), rtol=RTOL, atol=1e-4)
        np.testing.assert_allclose(ln.beta.grad, tb.grad.numpy(), rtol=RTOL, atol=1e-4)

    # --- 11.2 CrossEntropy: kayip degeri ile gradyan ayni fonksiyona ait olmali ---
    def test_crossentropy_value_matches_logsoftmax_at_extremes(self):
        logits = np.array([[1000.0, -1000.0], [3.0, 1.0], [-50.0, 60.0]], dtype=np.float32)
        target = np.array([1, 0, 0])
        loss = nv.CrossEntropyLoss()(nv.Tensor(logits), nv.Tensor(target))
        ref = torch.nn.functional.cross_entropy(_t64(logits), torch.tensor(target))
        # Dogru deger ~ (2000 + 0.127 + 110)/3; log(p+1e-12) ~27.6'da kirpar.
        self.assertAlmostEqual(loss.item(), ref.item(), delta=1e-3 * abs(ref.item()))

    def test_crossentropy_grad_vs_torch(self):
        logits = self.rng.standard_normal((5, 4)).astype(np.float32) * 3
        target = self.rng.integers(0, 4, 5)
        lt = nv.Tensor(logits, requires_grad=True)
        nv.CrossEntropyLoss()(lt, nv.Tensor(target)).backward()
        tl = _t64(logits, True)
        torch.nn.functional.cross_entropy(tl, torch.tensor(target)).backward()
        np.testing.assert_allclose(lt.grad, tl.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- 7.5 GELU: ileri tanh yaklasimi -> geri de tanh yaklasiminin turevi olmali ---
    def test_gelu_forward_backward_tanh_approximation(self):
        x = np.linspace(-6, 6, 241).astype(np.float32)
        xt = nv.Tensor(x, requires_grad=True)
        y = nv.GELU()(xt)
        y.sum().backward()
        tx = _t64(x, True)
        ty = torch.nn.functional.gelu(tx, approximate="tanh")
        ty.sum().backward()
        np.testing.assert_allclose(y.data, ty.detach().numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- C. Sayisal kararlilik ---
    def test_sigmoid_extreme_inputs_finite(self):
        x = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
        xt = nv.Tensor(x, requires_grad=True)
        y = nv.Sigmoid()(xt)
        y.sum().backward()
        self.assertTrue(np.all(np.isfinite(y.data)))
        self.assertTrue(np.all(np.isfinite(xt.grad)))
        np.testing.assert_allclose(y.data, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    # --- 10. MHSA: fan-out (X -> Q,K,V) ve 3D Linear dW batch-toplamasi ---
    def test_mhsa_full_chain_grad_vs_torch(self):
        B, S, D, H = 2, 5, 8, 2
        x = self.rng.standard_normal((B, S, D)).astype(np.float32)
        R = self.rng.standard_normal((B, S, D)).astype(np.float32)
        mha = nv.MultiHeadAttention(D, H)
        xt = nv.Tensor(x, requires_grad=True)
        (mha(xt) * nv.Tensor(R)).sum().backward()

        p = {n: _t64(getattr(mha, n).w.data, True) for n in ("q_proj", "k_proj", "v_proj", "out_proj")}
        b = {n: _t64(getattr(mha, n).bias.data, True) for n in p}
        tx = _t64(x, True)
        heads = lambda t: t.reshape(B, S, H, D // H).permute(0, 2, 1, 3)
        q, k, v = (heads(tx @ p[n] + b[n]) for n in ("q_proj", "k_proj", "v_proj"))
        a = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(D // H), dim=-1)
        o = (a @ v).permute(0, 2, 1, 3).reshape(B, S, D) @ p["out_proj"] + b["out_proj"]
        (o * _t64(R)).sum().backward()

        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=1e-3, atol=1e-5)
        for n in p:
            np.testing.assert_allclose(getattr(mha, n).w.grad, p[n].grad.numpy(), rtol=1e-3, atol=1e-5)
            np.testing.assert_allclose(getattr(mha, n).bias.grad, b[n].grad.numpy(), rtol=1e-3, atol=1e-5)

    # --- 13. Kuantizasyon kenar durumlari ---
    def test_quantize_zero_tensor_and_symmetric_range(self):
        q, s = nv.quantize_tensor(np.zeros((4, 4), dtype=np.float32))
        self.assertGreater(s, 0.0)
        self.assertTrue(np.all(q == 0))
        q, s = nv.quantize_tensor(self.rng.standard_normal(1000).astype(np.float32))
        # Simetrik form: q in [-127, 127]; -128 kullanilmaz
        self.assertGreaterEqual(int(q.min()), -127)
        self.assertLessEqual(int(q.max()), 127)

    # --- 5. Conv2d Groups > 1 Forward & Backward Paritesi ---
    def test_conv2d_groups_forward_backward_vs_torch(self):
        x = self.rng.standard_normal((2, 4, 6, 6)).astype(np.float32)
        w = self.rng.standard_normal((4, 2, 3, 3)).astype(np.float32)
        b = self.rng.standard_normal(4).astype(np.float32)
        c = nv.Conv2d(4, 4, 3, groups=2)
        c.w.data = w.copy()
        c.bias.data = b.copy()
        xt = nv.Tensor(x, requires_grad=True)
        c(xt).sum().backward()

        tx = _t64(x, True)
        tw = _t64(w, True)
        tb = _t64(b, True)
        torch.nn.functional.conv2d(tx, tw, tb, groups=2).sum().backward()

        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(c.w.grad, tw.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(c.bias.grad, tb.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- 2. DAG Fan-Out & Residual Baglanti Gradyan Toplamasi ---
    def test_residual_fan_out_accumulation(self):
        x = self.rng.standard_normal((3, 8)).astype(np.float32)
        w = self.rng.standard_normal((8, 8)).astype(np.float32)
        fc = nv.Linear(8, 8, bias=False)
        fc.w.data = w.copy()
        xt = nv.Tensor(x, requires_grad=True)
        y = xt + fc(xt)
        y.sum().backward()

        tx = _t64(x, True)
        tw = _t64(w, True)
        ty = tx + tx @ tw
        ty.sum().backward()

        np.testing.assert_allclose(xt.grad, tx.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- 2.3 & 3. Elemanter Operasyonlarda Unbroadcasting ---
    def test_unbroadcasting_elementwise_ops(self):
        a = self.rng.standard_normal((3, 4)).astype(np.float32)
        b = self.rng.standard_normal((1, 4)).astype(np.float32)
        at = nv.Tensor(a, requires_grad=True)
        bt = nv.Tensor(b, requires_grad=True)
        (at * bt).sum().backward()

        ta = _t64(a, True)
        tb = _t64(b, True)
        (ta * tb).sum().backward()

        np.testing.assert_allclose(at.grad, ta.grad.numpy(), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(bt.grad, tb.grad.numpy(), rtol=RTOL, atol=ATOL)

    # --- 11.2 CrossEntropy Yumusak Hedefler (Soft Targets / Label Smoothing) ---
    def test_crossentropy_soft_targets_vs_torch(self):
        logits = self.rng.standard_normal((4, 5)).astype(np.float32)
        targets = np.array([
            [0.8, 0.05, 0.05, 0.05, 0.05],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.05, 0.05, 0.8, 0.05, 0.05],
            [0.2, 0.2, 0.2, 0.2, 0.2]
        ], dtype=np.float32)
        lt = nv.Tensor(logits, requires_grad=True)
        nv.CrossEntropyLoss()(lt, nv.Tensor(targets)).backward()

        tl = _t64(logits, True)
        tt = _t64(targets, False)
        torch.nn.functional.cross_entropy(tl, tt).backward()

        np.testing.assert_allclose(lt.grad, tl.grad.numpy(), rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    unittest.main()
