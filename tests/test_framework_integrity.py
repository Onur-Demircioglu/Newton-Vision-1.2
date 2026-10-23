"""newton-vision Çerçeve Bütünlüğü ve Gelişmiş Özellik Testleri (tests/test_framework_integrity.py).

Test edilen özellikler:
1. no_grad() context manager ile gradyan üretimini dondurma
2. Sequential konteynırı ve model parametre hiyerarşisi
3. TensorDataset & DataLoader (mini-batching & shuffle)
4. PyTorch Linear ağırlıkları (out_features, in_features) için akıllı load_state_dict
5. CrossEntropyLoss (1D ve One-Hot hedefler, reduction='sum'/'mean')
"""

import unittest
import numpy as np
import newton_vision as nv


class TestFrameworkIntegrity(unittest.TestCase):

    def test_no_grad(self):
        """no_grad() içinde forward pass gradyan grafı oluşturmamalıdır."""
        x = nv.Tensor(np.random.randn(4, 8), requires_grad=True)
        linear = nv.Linear(8, 4)

        # Normalde requires_grad = True olmalı
        out_grad = linear(x)
        self.assertTrue(out_grad.requires_grad)
        self.assertGreater(len(out_grad._prev), 0)

        # with nv.no_grad(): altında requires_grad = False olmalı ve graph oluşmamalı
        with nv.no_grad():
            out_no_grad = linear(x)
            self.assertFalse(out_no_grad.requires_grad)
            self.assertEqual(len(out_no_grad._prev), 0)

    def test_sequential_container(self):
        """Sequential katman zincirleme ve özyineli parametre toplama."""
        model = nv.Sequential(
            nv.Linear(10, 20),
            nv.ReLU(),
            nv.Linear(20, 2)
        )

        params = model.parameters()
        # 2 adet Linear katman (w + bias = 2 + 2 = 4 parametre)
        self.assertEqual(len(params), 4)

        x = nv.Tensor(np.random.randn(5, 10))
        out = model(x)
        self.assertEqual(out.shape, (5, 2))

    def test_dataset_and_dataloader(self):
        """TensorDataset ve DataLoader mini-batch ve shuffle doğrulaması."""
        X_raw = np.random.randn(100, 10).astype(np.float32)
        y_raw = np.random.randint(0, 5, size=(100,)).astype(np.int64)

        dataset = nv.TensorDataset(X_raw, y_raw)
        self.assertEqual(len(dataset), 100)

        loader = nv.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)
        self.assertEqual(len(loader), 4)  # 32 + 32 + 32 + 4 = 100

        total_samples = 0
        for bx, by in loader:
            self.assertIsInstance(bx, nv.Tensor)
            self.assertIsInstance(by, nv.Tensor)
            total_samples += bx.shape[0]

        self.assertEqual(total_samples, 100)

    def test_smart_load_state_dict_auto_transpose(self):
        """PyTorch Linear (out, in) ağırlıklarının newton-vision (in, out) formatına otomatik dönüştürülmesi."""
        model = nv.Linear(in_features=16, out_features=8)
        
        # PyTorch formatında weight: (out_features=8, in_features=16)
        pytorch_weight = np.random.randn(8, 16).astype(np.float32)
        pytorch_bias = np.random.randn(8).astype(np.float32)

        state_dict = {
            "w": pytorch_weight,
            "bias": pytorch_bias
        }

        nv.load_state_dict(model, state_dict)

        # newton-vision w.data şekli (16, 8) olmalı ve pytorch_weight.T ile eşleşmeli
        self.assertEqual(model.w.shape, (16, 8))
        np.testing.assert_allclose(model.w.data, pytorch_weight.T)
        np.testing.assert_allclose(model.bias.data, pytorch_bias)

    def test_crossentropy_one_hot_and_reduction(self):
        """CrossEntropyLoss 1D ve One-Hot hedef desteği ile reduction seçenekleri."""
        logits = nv.Tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32), requires_grad=True)
        targets_1d = np.array([0, 1], dtype=np.int64)
        targets_one_hot = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        loss_mean = nv.gravity.CrossEntropyLoss(reduction="mean")
        l1 = loss_mean(logits, targets_1d)
        l2 = loss_mean(logits, targets_one_hot)

        # 1D ve One-Hot çıktısı neredeyse özdeş olmalı
        self.assertAlmostEqual(float(l1.data), float(l2.data), places=5)

        # reduction="sum" testi
        loss_sum = nv.gravity.CrossEntropyLoss(reduction="sum")
        l_sum = loss_sum(logits, targets_1d)
        self.assertAlmostEqual(float(l_sum.data), float(l1.data) * 2.0, places=5)

    def test_module_list_and_activation_module_registration(self):
        """ModuleList ve aktivasyon katmanlarinin Module hiyerarsisine tam entegrasyonu."""
        relu = nv.ReLU()
        self.assertIsInstance(relu, nv.Module)

        layers = nv.ModuleList([nv.Linear(4, 8), nv.ReLU(), nv.Linear(8, 2)])
        self.assertEqual(len(layers), 3)
        self.assertEqual(len(layers.parameters()), 4)

        x = nv.Tensor(np.random.randn(2, 4))
        out = x
        for layer in layers:
            out = layer(out)
        self.assertEqual(out.shape, (2, 2))

    def test_buffer_tracking_and_getattr(self):
        """register_buffer ve dinamik attribute yonetimi."""
        bn = nv.BatchNorm2d(8)
        self.assertIn("running_mean", bn._buffers)
    def test_tensor_item_and_numpy_and_len(self):
        """Tensor skaler donusum (.item()), ndarray erisimi (.numpy()) ve len() destegi."""
        scalar_t = nv.Tensor(3.14159)
        self.assertAlmostEqual(scalar_t.item(), 3.14159, places=5)
        self.assertIsInstance(scalar_t.item(), float)

        multi_t = nv.Tensor(np.ones((5, 10), dtype=np.float32))
        self.assertEqual(len(multi_t), 5)
        self.assertIsInstance(multi_t.numpy(), np.ndarray)
        self.assertEqual(multi_t.numpy().shape, (5, 10))

        with self.assertRaises(ValueError):
            multi_t.item()

    def test_root_namespace_loss_and_optim(self):
        """nv.CrossEntropyLoss, nv.MSELoss, nv.Adam ve nv.SGD direkt root export erisimi."""
        crit = nv.CrossEntropyLoss()
        self.assertIsInstance(crit, nv.Module)

        mse = nv.MSELoss()
        self.assertIsInstance(mse, nv.Module)

        model = nv.Linear(4, 2)
        opt_adam = nv.Adam(model.parameters(), lr=0.01)
        opt_sgd = nv.SGD(model.parameters(), lr=0.01)
        self.assertIsNotNone(opt_adam)
        self.assertIsNotNone(opt_sgd)

    def test_dropout_train_eval_and_backward(self):
        """Dropout egitim, cikarim (eval) ve ters turev (backward) dogrulamasi."""
        drop = nv.Dropout(p=0.4)
        x = nv.Tensor(np.ones((10, 20), dtype=np.float32), requires_grad=True)

        # Train modunda bazi noronlar sifirlanmali
        drop.train(True)
        out_train = drop(x)
        nonzero = np.count_nonzero(out_train.data)
        self.assertLess(nonzero, 200)
        self.assertGreater(nonzero, 50)

        out_train.sum().backward()
        self.assertEqual(np.count_nonzero(x.grad), nonzero)

        # Eval modunda birebir giris donmeli
        drop.eval()
        x2 = nv.Tensor(np.ones((5, 5), dtype=np.float32))
        out_eval = drop(x2)
        np.testing.assert_array_equal(out_eval.data, x2.data)

    def test_weight_decay(self):
        """Adam ve SGD optimizatorlerinde weight_decay (L2 regularization) dogrulamasi."""
        w = nv.Tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        opt = nv.Adam([w], lr=0.01, weight_decay=0.1)
        zero_loss = (w * 0.0).sum()
        zero_loss.backward()
        opt.step()
        self.assertLess(w.data[0, 0], 1.0)


if __name__ == "__main__":
    unittest.main()


