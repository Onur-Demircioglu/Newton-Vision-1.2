"""Gerçek ONNX dışa aktarımı: model zoo + yeni katmanlar onnxruntime ile doğrulanır.

onnx / onnxruntime kurulu değilse testler açıkça SKIP edilir.
"""

from __future__ import annotations
import os
import tempfile
import unittest
import numpy as np
import newton_vision as nv
from newton_vision.tensor import Tensor

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


@unittest.skipUnless(HAS_ONNX, "onnx / onnxruntime kurulu değil")
class TestOnnxExport(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _export_and_compare(self, model, x):
        path = os.path.join(self.tmp.name, "m.onnx")
        nv.export_onnx(model, x, path)          # verify=True: iki girdiyle iç doğrulama
        onnx.checker.check_model(onnx.load(path))
        # Bağımsız karşılaştırma: farklı bir girdi
        rng = np.random.default_rng(42)
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            other = rng.integers(0, int(np.max(x)) + 1, np.shape(x)).astype(np.int64)
        else:
            other = rng.standard_normal(np.shape(x)).astype(np.float32)
        got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": other})[0]
        model.eval()
        with nv.no_grad():
            expected = model(Tensor(other)).data
        np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-3)
        return onnx.load(path)

    def test_resnet18(self):
        proto = self._export_and_compare(nv.models.resnet18(num_classes=10), np.random.randn(1, 3, 224, 224).astype(np.float32))
        ops = {n.op_type for n in proto.graph.node}
        self.assertTrue({"Conv", "BatchNormalization", "Add", "MaxPool", "MatMul"} <= ops)

    def test_vit_tiny(self):
        self._export_and_compare(nv.models.vit_tiny(num_classes=5, img_size=32, patch_size=8),
                                 np.random.randn(2, 3, 32, 32).astype(np.float32))

    def test_mobilenet_v2(self):
        self._export_and_compare(nv.models.mobilenet_v2(num_classes=5), np.random.randn(1, 3, 224, 224).astype(np.float32))

    def test_new_layers(self):
        seq = nv.Sequential(nv.Conv1d(4, 8, 3, padding=1), nv.BatchNorm1d(8), nv.LeakyReLU(0.1),
                            nv.Upsample(2, mode="linear"), nv.Flatten(), nv.Dropout(0.3), nv.Linear(128, 3), nv.Sigmoid())
        self._export_and_compare(seq, np.random.randn(2, 4, 8).astype(np.float32))
        img = nv.Sequential(nv.Upsample(2, mode="bilinear", align_corners=True), nv.Upsample(3), nv.Conv2d(2, 2, 3),
                            nv.Tanh(), nv.MaxPool2d(3, 2, padding=1), nv.AvgPool2d(2))
        self._export_and_compare(img, np.random.randn(1, 2, 5, 5).astype(np.float32))
        emb = nv.Sequential(nv.Embedding(20, 6), nv.LayerNorm(6), nv.GELU())
        self._export_and_compare(emb, np.random.randint(0, 20, (3, 5)))

    def test_graph_breaking_model_is_rejected(self):
        class Broken(nv.Module):
            def __init__(self):
                super().__init__()
                self.c = nv.Conv2d(1, 2, 3)
                self.f = nv.Linear(32, 3)

            def forward(self, x):
                h = self.c(x)
                return self.f(nv.Tensor(h.data.reshape(h.shape[0], -1)))   # girdiye bağlı değer sabitlenir

        with self.assertRaises(RuntimeError):
            nv.export_onnx(Broken(), np.random.randn(1, 1, 6, 6).astype(np.float32), os.path.join(self.tmp.name, "b.onnx"))

    def test_unsupported_layer_raises_and_state_is_restored(self):
        q = nv.Sequential(nv.Linear(4, 3))
        nv.quantize_model(q)
        add_before = Tensor.__add__
        with self.assertRaises(NotImplementedError):
            nv.export_onnx(q, np.zeros((1, 4), np.float32), os.path.join(self.tmp.name, "q.onnx"))
        self.assertIs(Tensor.__add__, add_before, "izleme yamaları geri alınmalı")
        m = nv.models.resnet18(num_classes=2)
        m.train()
        nv.export_onnx(m, np.random.randn(1, 3, 224, 224).astype(np.float32), os.path.join(self.tmp.name, "r.onnx"))
        self.assertTrue(all(mod.training for mod in m.modules()), "eğitim modu geri yüklenmeli")


if __name__ == "__main__":
    unittest.main()
