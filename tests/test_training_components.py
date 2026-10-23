"""Eğitim bileşenleri: yeni katmanlar, kayıplar, optimizer durumu, scheduler, gradyan kırpma.

Her bileşen forward ve backward olarak PyTorch'a (float64 referans) karşı doğrulanır.
torch kurulu değilse PyTorch karşılaştırmaları açıkça SKIP edilir.
"""

from __future__ import annotations
import unittest
import numpy as np
import newton_vision as nv

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ATOL = 1e-5


def _t(a, grad=True):
    return torch.tensor(np.asarray(a, dtype=np.float64), requires_grad=grad)


@unittest.skipUnless(HAS_TORCH, "torch kurulu değil")
class TestNewLayers(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        torch.manual_seed(0)

    def _check_layer(self, nv_layer, torch_fn, x, torch_params=()):
        """nv_layer(x) ile torch_fn(x, *params) forward ve x/param gradyanlarını karşılaştırır."""
        xn = nv.Tensor(x, requires_grad=True)
        yn = nv_layer(xn)
        g = np.random.randn(*yn.shape).astype(np.float32)
        (yn * g).sum().backward()

        xt = _t(x)
        pt = [_t(p.data) for p, _ in torch_params]
        yt = torch_fn(xt, *pt)
        (yt * _t(g, grad=False)).sum().backward()

        np.testing.assert_allclose(yn.data, yt.detach().numpy(), atol=ATOL, rtol=1e-4)
        np.testing.assert_allclose(xn.grad, xt.grad.numpy(), atol=ATOL, rtol=1e-4)
        for (p, name), ptt in zip(torch_params, pt):
            np.testing.assert_allclose(p.grad, ptt.grad.numpy(), atol=ATOL, rtol=1e-4, err_msg=name)

    def test_flatten_preserves_graph(self):
        conv = nv.Conv2d(1, 2, 3)
        out = nv.Flatten()(conv(nv.Tensor(np.random.randn(2, 1, 5, 5), requires_grad=True)))
        self.assertEqual(out.shape, (2, 18))
        out.sum().backward()
        self.assertGreater(np.abs(conv.w.grad).sum(), 0.0, "Flatten grafı koparmamalı")
        self.assertEqual(nv.Flatten(0, 1)(np.zeros((2, 3, 4))).shape, (6, 4))

    def test_conv1d(self):
        for stride, padding, dilation, groups in [(1, 0, 1, 1), (2, 2, 1, 1), (1, 1, 2, 2), (1, 3, 1, 4)]:
            layer = nv.Conv1d(4, 8, 3, stride=stride, padding=padding, dilation=dilation, groups=groups)
            x = np.random.randn(2, 4, 11).astype(np.float32)
            self._check_layer(
                layer,
                lambda xt, w, b: torch.nn.functional.conv1d(xt, w, b, stride, padding, dilation, groups),
                x, [(layer.w, "w"), (layer.bias, "bias")])

    def test_batchnorm1d_train_and_eval(self):
        for shape in [(8, 5), (4, 5, 7)]:
            bn = nv.BatchNorm1d(5)
            bn.gamma.data = (np.random.rand(5) + 0.5).astype(np.float32)
            bn.beta.data = np.random.randn(5).astype(np.float32)
            x = np.random.randn(*shape).astype(np.float32)
            rm = torch.zeros(5, dtype=torch.float64)
            rv = torch.ones(5, dtype=torch.float64)
            self._check_layer(
                bn, lambda xt, g, b: torch.nn.functional.batch_norm(xt, rm, rv, g, b, training=True, eps=1e-5),
                x, [(bn.gamma, "gamma"), (bn.beta, "beta")])
            np.testing.assert_allclose(bn.running_mean, rm.numpy(), atol=ATOL)
            np.testing.assert_allclose(bn.running_var, rv.numpy(), atol=ATOL)
            bn.eval()
            ye = bn(x).data
            yt = torch.nn.functional.batch_norm(_t(x, False), rm, rv, _t(bn.gamma.data, False), _t(bn.beta.data, False), training=False)
            np.testing.assert_allclose(ye, yt.numpy(), atol=ATOL)

    def test_upsample_nearest(self):
        for shape in [(2, 3, 4), (2, 3, 4, 5)]:
            self._check_layer(nv.Upsample(scale_factor=2), lambda xt: torch.nn.functional.interpolate(xt, scale_factor=2, mode="nearest"),
                              np.random.randn(*shape).astype(np.float32))
        with self.assertRaises(ValueError):
            nv.Upsample(2, mode="bicubic")
        with self.assertRaises(ValueError):
            nv.Upsample(2, mode="nearest", align_corners=True)

    def test_upsample_linear_and_bilinear(self):
        for scale in (2, 3):
            for ac in (False, True):
                self._check_layer(
                    nv.Upsample(scale, mode="bilinear", align_corners=ac),
                    lambda xt: torch.nn.functional.interpolate(xt, scale_factor=scale, mode="bilinear", align_corners=ac),
                    np.random.randn(2, 3, 4, 5).astype(np.float32))
                self._check_layer(
                    nv.Upsample(scale, mode="linear", align_corners=ac),
                    lambda xt: torch.nn.functional.interpolate(xt, scale_factor=scale, mode="linear", align_corners=ac),
                    np.random.randn(2, 3, 6).astype(np.float32))
        with self.assertRaises(ValueError):
            nv.Upsample(2, mode="bilinear")(np.zeros((1, 1, 4)))

    def test_embedding(self):
        emb = nv.Embedding(10, 4, padding_idx=0)
        self.assertTrue(np.all(emb.weight.data[0] == 0))
        idx = np.array([[0, 3, 3], [9, 1, 0]])
        out = emb(idx)
        g = np.random.randn(*out.shape).astype(np.float32)
        (out * g).sum().backward()
        wt = _t(emb.weight.data)
        yt = torch.nn.functional.embedding(torch.tensor(idx), wt, padding_idx=0)
        (yt * _t(g, False)).sum().backward()
        np.testing.assert_allclose(out.data, yt.detach().numpy(), atol=ATOL)
        np.testing.assert_allclose(emb.weight.grad, wt.grad.numpy(), atol=ATOL)
        with self.assertRaises(IndexError):
            emb(np.array([10]))
        with self.assertRaises(TypeError):
            emb(np.array([1.5]))


@unittest.skipUnless(HAS_TORCH, "torch kurulu değil")
class TestNewLosses(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def _check_loss(self, nv_loss, torch_loss, pred, target):
        pn = nv.Tensor(pred, requires_grad=True)
        ln = nv_loss(pn, target)
        g = np.random.randn(*ln.shape).astype(np.float32) if ln.data.ndim else np.float32(1.0)
        (ln * g).sum().backward() if ln.data.ndim else ln.backward()
        pt = _t(pred)
        lt = torch_loss(pt, _t(target, False))
        (lt * _t(g, False)).sum().backward()
        np.testing.assert_allclose(ln.data, lt.detach().numpy(), atol=ATOL, rtol=1e-4)
        np.testing.assert_allclose(pn.grad, pt.grad.numpy(), atol=ATOL, rtol=1e-4)

    def test_l1_loss(self):
        for red in ("mean", "sum", "none"):
            self._check_loss(nv.L1Loss(red), torch.nn.L1Loss(reduction=red),
                             np.random.randn(4, 3).astype(np.float32), np.random.randn(4, 3).astype(np.float32))

    def test_bce_loss(self):
        for red in ("mean", "sum", "none"):
            p = np.random.uniform(0.05, 0.95, (4, 3)).astype(np.float32)
            y = (np.random.rand(4, 3) > 0.5).astype(np.float32)
            self._check_loss(nv.BCELoss(red), torch.nn.BCELoss(reduction=red), p, y)

    def test_bce_with_logits_loss(self):
        for red in ("mean", "sum", "none"):
            x = (np.random.randn(4, 3) * 5).astype(np.float32)
            y = np.random.rand(4, 3).astype(np.float32)
            self._check_loss(nv.BCEWithLogitsLoss(red), torch.nn.BCEWithLogitsLoss(reduction=red), x, y)
        # Büyük logit'lerde sayısal kararlılık (inf/nan üretmemeli)
        ln = nv.BCEWithLogitsLoss()(nv.Tensor(np.array([1000.0, -1000.0]), requires_grad=True), np.array([0.0, 1.0]))
        self.assertTrue(np.isfinite(ln.item()))
        self.assertAlmostEqual(ln.item(), 1000.0, places=3)


@unittest.skipUnless(HAS_TORCH, "torch kurulu değil")
class TestOptimizationUtilities(unittest.TestCase):

    def _pair(self):
        """Aynı ağırlıklarla bir nv ve bir torch Linear modeli."""
        np.random.seed(2)
        nv_lin = nv.Linear(5, 3)
        t_lin = torch.nn.Linear(5, 3).double()
        with torch.no_grad():
            t_lin.weight.copy_(torch.tensor(nv_lin.w.data.T))
            t_lin.bias.copy_(torch.tensor(nv_lin.bias.data))
        return nv_lin, t_lin

    def _step(self, nv_lin, t_lin, nv_opt, t_opt, x, y, clip=None):
        loss = nv.MSELoss()(nv_lin(x), y)
        nv_opt.zero_grad(); loss.backward()
        tl = torch.nn.functional.mse_loss(t_lin(_t(x, False)), _t(y, False))
        t_opt.zero_grad(); tl.backward()
        if clip is not None:
            nn_norm = nv.clip_grad_norm_(nv_lin.parameters(), clip)
            tn_norm = torch.nn.utils.clip_grad_norm_(t_lin.parameters(), clip)
            self.assertAlmostEqual(nn_norm, float(tn_norm), places=4)
        nv_opt.step(); t_opt.step()

    def _assert_same(self, nv_lin, t_lin):
        np.testing.assert_allclose(nv_lin.w.data.T, t_lin.weight.detach().numpy(), atol=1e-5)
        np.testing.assert_allclose(nv_lin.bias.data, t_lin.bias.detach().numpy(), atol=1e-5)

    def test_clip_grad_norm_matches_torch(self):
        nv_lin, t_lin = self._pair()
        x = np.random.randn(8, 5).astype(np.float32) * 10
        y = np.random.randn(8, 3).astype(np.float32)
        nv_opt, t_opt = nv.SGD(nv_lin.parameters(), lr=0.01), torch.optim.SGD(t_lin.parameters(), lr=0.01)
        for _ in range(3):
            self._step(nv_lin, t_lin, nv_opt, t_opt, x, y, clip=0.5)
        self._assert_same(nv_lin, t_lin)

    def test_schedulers_match_torch(self):
        for make in [lambda o, t: (nv.StepLR(o, step_size=3, gamma=0.5), torch.optim.lr_scheduler.StepLR(t, step_size=3, gamma=0.5)),
                     lambda o, t: (nv.CosineAnnealingLR(o, T_max=5, eta_min=0.001), torch.optim.lr_scheduler.CosineAnnealingLR(t, T_max=5, eta_min=0.001))]:
            nv_lin, t_lin = self._pair()
            nv_opt, t_opt = nv.Adam(nv_lin.parameters(), lr=0.1), torch.optim.Adam(t_lin.parameters(), lr=0.1)
            nv_s, t_s = make(nv_opt, t_opt)
            for _ in range(12):
                self.assertAlmostEqual(nv_opt.lr, t_opt.param_groups[0]["lr"], places=10)
                self.assertAlmostEqual(nv_s.get_last_lr()[0], t_s.get_last_lr()[0], places=10)
                nv_opt.step(); t_opt.step()  # gradyan yok: torch uyarısını önlemek için önce optimizer
                nv_s.step(); t_s.step()

    def test_adamw_matches_torch(self):
        nv_lin, t_lin = self._pair()
        x = np.random.randn(8, 5).astype(np.float32)
        y = np.random.randn(8, 3).astype(np.float32)
        nv_opt = nv.AdamW(nv_lin.parameters(), lr=0.05, weight_decay=0.1)
        t_opt = torch.optim.AdamW(t_lin.parameters(), lr=0.05, weight_decay=0.1)
        for _ in range(10):
            self._step(nv_lin, t_lin, nv_opt, t_opt, x, y)
        self._assert_same(nv_lin, t_lin)
        # AdamW, aynı weight_decay ile Adam'dan farklı sonuç vermeli (ayrık azaltma)
        a, _ = self._pair(); b, _ = self._pair()
        oa, ob = nv.Adam(a.parameters(), lr=0.05, weight_decay=0.1), nv.AdamW(b.parameters(), lr=0.05, weight_decay=0.1)
        for m, o in ((a, oa), (b, ob)):
            for _ in range(5):
                loss = nv.MSELoss()(m(x), y); o.zero_grad(); loss.backward(); o.step()
        self.assertFalse(np.allclose(a.w.data, b.w.data))
        with self.assertRaises(ValueError):
            nv.Adam(b.parameters()).load_state_dict(ob.state_dict())

    def test_optimizer_state_dict_resume(self):
        """Eğitimi kesip optimizer + model durumundan devam ettirmek, kesintisiz eğitimle aynı sonucu vermeli."""
        x = np.random.randn(8, 5).astype(np.float32)
        y = np.random.randn(8, 3).astype(np.float32)
        for make_opt in [lambda p: nv.Adam(p, lr=0.05, weight_decay=0.01), lambda p: nv.AdamW(p, lr=0.05), lambda p: nv.SGD(p, lr=0.05, momentum=0.9)]:
            np.random.seed(3)
            ref = nv.Linear(5, 3)
            start = ref.state_dict()
            ref_opt = make_opt(ref.parameters())
            for _ in range(6):
                loss = nv.MSELoss()(ref(x), y); ref_opt.zero_grad(); loss.backward(); ref_opt.step()

            a = nv.Linear(5, 3); a.load_state_dict(start); a_opt = make_opt(a.parameters())
            for _ in range(3):
                loss = nv.MSELoss()(a(x), y); a_opt.zero_grad(); loss.backward(); a_opt.step()
            model_ckpt, opt_ckpt = a.state_dict(), a_opt.state_dict()

            b = nv.Linear(5, 3); b.load_state_dict(model_ckpt)
            b_opt = make_opt(b.parameters()); b_opt.load_state_dict(opt_ckpt)
            for _ in range(3):
                loss = nv.MSELoss()(b(x), y); b_opt.zero_grad(); loss.backward(); b_opt.step()

            np.testing.assert_allclose(b.w.data, ref.w.data, atol=1e-6)
            np.testing.assert_allclose(b.bias.data, ref.bias.data, atol=1e-6)

    def test_optimizer_state_dict_rejects_mismatch(self):
        opt = nv.Adam(nv.Linear(5, 3).parameters())
        with self.assertRaises(ValueError):
            nv.Adam(nv.Linear(4, 3).parameters()).load_state_dict(opt.state_dict())
        with self.assertRaises(ValueError):
            nv.SGD(nv.Linear(5, 3).parameters()).load_state_dict(opt.state_dict())


class TestTransforms(unittest.TestCase):

    def test_random_horizontal_flip(self):
        x = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        np.testing.assert_array_equal(nv.transforms.RandomHorizontalFlip(p=1.0)(x), x[:, :, ::-1])
        np.testing.assert_array_equal(nv.transforms.RandomHorizontalFlip(p=0.0)(x), x)

    def test_random_crop_with_padding(self):
        np.random.seed(0)
        x = np.random.randn(2, 8, 8).astype(np.float32)
        crop = nv.transforms.RandomCrop(8, padding=2)
        seen_shift = False
        for _ in range(20):
            y = crop(x)
            self.assertEqual(y.shape, (2, 8, 8))
            # Kırpılan pencere, sıfır dolgulu girdinin bir alt penceresi olmalı
            padded = np.pad(x, ((0, 0), (2, 2), (2, 2)))
            found = any(np.array_equal(padded[:, i:i + 8, j:j + 8], y) for i in range(5) for j in range(5))
            self.assertTrue(found)
            seen_shift |= not np.array_equal(y, x)
        self.assertTrue(seen_shift)

    def test_normalize_and_compose(self):
        x = np.ones((2, 2, 2), dtype=np.float32)
        t = nv.transforms.Compose([nv.transforms.Normalize(mean=[0.5, 1.0], std=[0.5, 2.0])])
        y = t(x)
        np.testing.assert_allclose(y[0], 1.0)
        np.testing.assert_allclose(y[1], 0.0)

    def test_tensor_dataset_transform_applies_per_sample(self):
        x = np.arange(4 * 1 * 2 * 2, dtype=np.float32).reshape(4, 1, 2, 2)
        y = np.arange(4)
        ds = nv.TensorDataset(x, y, transform=nv.transforms.RandomHorizontalFlip(p=1.0))
        bx, by = next(iter(nv.DataLoader(ds, batch_size=4)))
        np.testing.assert_array_equal(bx.data, x[:, :, :, ::-1])
        np.testing.assert_array_equal(by.data, y)
        # Dönüşüm orijinal veriyi değiştirmemeli
        np.testing.assert_array_equal(ds.tensors[0].data, x)


class TestTokenizer(unittest.TestCase):

    def test_word_vocab_is_deterministic_and_encodes(self):
        texts = ["Merhaba dünya!", "merhaba newton", "dünya dünya"]
        tok = nv.text.Tokenizer(level="word").fit(texts)
        # sıklık: dünya=3, merhaba=2, sonra alfabetik: !, newton
        self.assertEqual(tok.id_to_token, ["<pad>", "<unk>", "dünya", "merhaba", "!", "newton"])
        self.assertEqual(tok.encode("Merhaba yabancı dünya"), [3, 1, 2])
        self.assertEqual(tok.decode([3, 1, 2, 0]), "merhaba dünya")

    def test_char_level_and_batch_padding(self):
        tok = nv.text.Tokenizer(level="char", lower=False).fit(["abca"])
        batch = tok.encode_batch(["ab", "abcab"], max_len=4)
        self.assertEqual(batch.dtype, np.int64)
        self.assertEqual(batch.shape, (2, 4))
        self.assertEqual(batch[0, 2:].tolist(), [tok.pad_id, tok.pad_id])
        self.assertEqual(tok.decode(batch[1]), "abca")

    def test_min_freq_max_size_and_state_roundtrip(self):
        tok = nv.text.Tokenizer(min_freq=2, max_size=3).fit(["a a b b b c"])
        self.assertEqual(tok.id_to_token, ["<pad>", "<unk>", "b"])
        tok2 = nv.text.Tokenizer().load_state_dict(tok.state_dict())
        self.assertEqual(tok2.encode("a b c"), tok.encode("a b c"))
        with self.assertRaises(IndexError):
            tok.decode([99])

    def test_tokenizer_feeds_embedding(self):
        tok = nv.text.Tokenizer().fit(["iyi film", "kötü film"])
        ids = tok.encode_batch(["iyi film", "kötü"], max_len=3)
        emb = nv.Embedding(tok.vocab_size, 4, padding_idx=tok.pad_id)
        out = emb(ids)
        self.assertEqual(out.shape, (2, 3, 4))
        np.testing.assert_array_equal(out.data[1, 1:], 0.0)  # <pad> satırı sıfır


class TestEndToEndTraining(unittest.TestCase):
    """Yeni bileşenlerin gerçek bir eğitim döngüsünde birlikte çalıştığını, görülmemiş veride doğrular."""

    def test_image_pipeline_learns_and_generalizes(self):
        # Etiket: üst yarı alt yarıdan parlak mı? Yatay çevirme etiketi korur.
        np.random.seed(0)

        def make(n):
            x = np.random.randn(n, 1, 8, 8).astype(np.float32) * 0.5
            y = np.random.randint(0, 2, n)
            x[y == 1, :, :4, :] += 1.0
            x[y == 0, :, 4:, :] += 1.0
            return x, y

        x_tr, y_tr = make(256)
        x_te, y_te = make(200)
        model = nv.Sequential(
            nv.Conv2d(1, 4, 3, padding=1), nv.BatchNorm2d(4), nv.ReLU(), nv.MaxPool2d(2),
            nv.Flatten(), nv.Linear(64, 16), nv.BatchNorm1d(16), nv.ReLU(), nv.Dropout(0.2), nv.Linear(16, 2),
        )
        opt = nv.Adam(model.parameters(), lr=0.01)
        sched = nv.StepLR(opt, step_size=5, gamma=0.5)
        loader = nv.DataLoader(nv.TensorDataset(x_tr, y_tr, transform=nv.transforms.RandomHorizontalFlip()), batch_size=32, shuffle=True)
        for _ in range(10):
            model.train()
            for bx, by in loader:
                loss = nv.CrossEntropyLoss()(model(bx), by)
                opt.zero_grad(); loss.backward()
                nv.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            sched.step()
        model.eval()
        with nv.no_grad():
            acc = np.mean(model(x_te).data.argmax(1) == y_te)
        self.assertGreater(acc, 0.95, f"Görülmemiş veride doğruluk düşük: {acc:.3f}")
        self.assertAlmostEqual(opt.lr, 0.01 * 0.25)

    def test_sequence_pipeline_embedding_conv1d(self):
        # Etiket: dizide 7 numaralı token var mı?
        np.random.seed(1)

        def make(n):
            seq = np.random.randint(0, 7, (n, 12))
            y = np.random.randint(0, 2, n)
            pos = np.random.randint(0, 12, n)
            seq[y == 1, pos[y == 1]] = 7
            return seq, y

        s_tr, y_tr = make(400)
        s_te, y_te = make(200)
        emb, conv, up = nv.Embedding(8, 6), nv.Conv1d(6, 8, 3, padding=1), nv.Upsample(scale_factor=2)
        head = nv.Linear(8, 1)
        params = emb.parameters() + conv.parameters() + head.parameters()
        opt = nv.Adam(params, lr=0.02)
        loss_fn = nv.BCEWithLogitsLoss()

        def forward(seq):
            h = emb(seq).transpose(0, 2, 1)          # (N, L, D) -> (N, D, L)
            h = up(nv.ReLU()(conv(h)))               # (N, 8, 2L)
            return head(h.max(axis=2)).reshape(-1)   # global max pooling -> logit

        for _ in range(60):
            idx = np.random.choice(len(s_tr), 64, replace=False)
            loss = loss_fn(forward(s_tr[idx]), y_tr[idx].astype(np.float32))
            opt.zero_grad(); loss.backward(); opt.step()
        with nv.no_grad():
            acc = np.mean((forward(s_te).data > 0) == y_te)
        self.assertGreater(acc, 0.95, f"Dizi görevinde doğruluk düşük: {acc:.3f}")


class TestAutogradRobustness(unittest.TestCase):

    def test_long_graph_backward_no_recursion_error(self):
        x = nv.Tensor(np.ones(3), requires_grad=True)
        y = x
        for _ in range(5000):
            y = y * 1.0
        y.sum().backward()
        np.testing.assert_allclose(x.grad, np.ones(3))

    def test_device_transfer_is_differentiable(self):
        x = nv.Tensor(np.array([1.0, 2.0]), requires_grad=True)
        y = x * 3.0
        z = nv.Tensor(y.data, device="cpu")  # aynı cihaz: kendisi döner
        self.assertIs(y.to("cpu"), y)
        w = y.to("cpu") * 2.0
        w.sum().backward()
        np.testing.assert_allclose(x.grad, [6.0, 6.0])
        self.assertFalse(z.requires_grad)


if __name__ == "__main__":
    unittest.main()
