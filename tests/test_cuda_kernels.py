"""CUDA çekirdekleri (conv2d, relu, maxpool2d forward) CPU yoluyla aynı sonucu vermeli.

Yalnızca CUDA ile derlenmiş bir _core ve CUDA cihazı varsa çalışır; aksi halde açıkça SKIP edilir.
Backward her zaman CPU'da hesaplanır; CUDA cihazındaki tensörlerin gradyanları da CPU ile karşılaştırılır.
"""

from __future__ import annotations
import unittest
import numpy as np
import newton_vision as nv
import newton_vision._core as _core

HAS_CUDA = _core.has_cuda()
ATOL, RTOL = 1e-4, 1e-4


@unittest.skipUnless(HAS_CUDA, "CUDA build'i veya CUDA cihazı yok")
class TestCudaKernels(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_conv2d_matches_cpu(self):
        # (N, C_in, H, W, C_out, k, stride, padding, dilation, groups, bias)
        configs = [
            (2, 3, 9, 9, 4, 3, 1, 1, 1, 1, True),
            (1, 4, 11, 7, 6, 3, 2, 0, 1, 1, False),
            (2, 8, 10, 10, 8, 3, 1, 2, 2, 1, True),     # dilation
            (2, 8, 10, 10, 12, 3, 1, 1, 1, 4, True),    # gruplu
            (2, 16, 12, 12, 16, 3, 2, 1, 1, 16, False), # depthwise
            (1, 3, 8, 8, 5, 1, 1, 0, 1, 1, True),       # 1x1
            (70, 1000, 1, 1, 1000, 1, 1, 0, 1, 1, True),  # N*C_out = 70000 > 65535 (eski grid.z sınırı)
        ]
        for N, C, H, W, Co, k, s, p, d, g, b in configs:
            with self.subTest(cfg=(N, C, H, W, Co, k, s, p, d, g, b)):
                x = np.random.randn(N, C, H, W).astype(np.float32)
                w = np.random.randn(Co, C // g, k, k).astype(np.float32)
                bias = np.random.randn(Co).astype(np.float32) if b else None
                y_cpu = _core.conv2d(x, w, bias, s, p, d, g, False)
                y_gpu = _core.conv2d(x, w, bias, s, p, d, g, True)
                self.assertEqual(y_cpu.shape, y_gpu.shape)
                np.testing.assert_allclose(y_gpu, y_cpu, atol=ATOL, rtol=RTOL)

    def test_relu_matches_cpu(self):
        x = np.random.randn(3, 5, 7, 11).astype(np.float32)
        x[0, 0, 0, :3] = 0.0
        np.testing.assert_array_equal(_core.relu(x, True), _core.relu(x, False))

    def test_maxpool2d_matches_cpu(self):
        for (k, s) in [(2, 2), (3, 2), (3, 1)]:
            with self.subTest(k=k, s=s):
                x = np.random.randn(2, 4, 9, 10).astype(np.float32)
                np.testing.assert_array_equal(_core.maxpool2d(x, k, s, True), _core.maxpool2d(x, k, s, False))
        # Python tarafının -inf dolgusu CUDA yolunda da doğru çalışmalı
        x = np.random.randn(2, 3, 7, 7).astype(np.float32)
        y_cpu = nv.MaxPool2d(3, 2, padding=1)(nv.Tensor(x)).data
        y_gpu = nv.MaxPool2d(3, 2, padding=1)(nv.Tensor(x).cuda()).data
        np.testing.assert_array_equal(y_gpu, y_cpu)

    def test_layer_forward_and_gradients_on_cuda_tensors(self):
        np.random.seed(1)
        x = np.random.randn(2, 3, 12, 12).astype(np.float32)
        results = {}
        for device in ("cpu", "cuda"):
            np.random.seed(2)
            conv = nv.Conv2d(3, 6, 3, stride=1, padding=1).to(device)
            pool, relu = nv.MaxPool2d(2), nv.ReLU()
            xt = nv.Tensor(x, requires_grad=True)
            inp = xt.to(device)
            out = pool(relu(conv(inp)))
            self.assertEqual(out.device, device)
            (out * out).sum().backward()
            results[device] = (out.data, xt.grad.copy(), conv.w.grad.copy(), conv.bias.grad.copy())
        for a, b in zip(results["cpu"], results["cuda"]):
            np.testing.assert_allclose(b, a, atol=ATOL, rtol=RTOL)

    def test_full_models_match_cpu(self):
        for name, make in [("resnet18", lambda: nv.models.resnet18(num_classes=10)),
                           ("mobilenet_v2", lambda: nv.models.mobilenet_v2(num_classes=10))]:
            with self.subTest(model=name):
                np.random.seed(3)
                model = make()
                model.eval()
                x = np.random.randn(1, 3, 224, 224).astype(np.float32)
                with nv.no_grad():
                    y_cpu = model(nv.Tensor(x)).data
                    model.to("cuda")
                    y_gpu = model(nv.Tensor(x).cuda()).data
                np.testing.assert_allclose(y_gpu, y_cpu, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
