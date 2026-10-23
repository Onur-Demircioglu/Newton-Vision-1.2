"""newton-vision v2.0: AVX2 SIMD & INT8 Kuantizasyon Engine Testleri."""

import unittest
import numpy as np
import newton_vision as nv


class TestQuantization(unittest.TestCase):

    def test_tensor_quantization(self):
        print("\n--- Tensör FP32 -> INT8 Kuantizasyon Testi ---")
        fp32_data = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        q_int8, scale = nv.quantize_tensor(fp32_data)

        self.assertEqual(q_int8.dtype, np.int8)
        self.assertEqual(q_int8[0], -127)
        self.assertEqual(q_int8[2], 0)
        self.assertEqual(q_int8[4], 127)

        # Dequantize back to FP32
        restored_fp32 = nv.dequantize_tensor(q_int8, scale)
        max_error = float(np.max(np.abs(restored_fp32 - fp32_data)))
        print(f"  [OK] Kuantizasyon ve Geri Donusturme Maksimum Hatasi: {max_error:.6f}")
        self.assertLess(max_error, 1e-2)

    def test_per_channel_quantization_nonzero_axis(self):
        fp32_data = np.random.randn(2, 3, 4).astype(np.float32)
        q_int8, scales = nv.quantize_tensor_per_channel(fp32_data, axis=1)

        self.assertEqual(q_int8.shape, fp32_data.shape)
        self.assertEqual(q_int8.dtype, np.int8)
        self.assertEqual(scales.shape, (3,))

    def test_quantized_linear(self):
        print("\n--- QuantizedLinear INT8 Katmani Cikarim Testi ---")
        q_linear = nv.QuantizedLinear(in_features=128, out_features=64)

        # Create random input tensor
        x_data = np.random.randn(4, 128).astype(np.float32)
        x = nv.Tensor(x_data, requires_grad=False)

        out = q_linear(x)
        self.assertEqual(out.data.shape, (4, 64))
        print(f"  [OK] QuantizedLinear INT8 Cikti Boyutu: {out.data.shape}")

    def test_model_quantization(self):
        print("\n--- Model Bazli INT8 Kuantizasyon & Bellek Kucultme Testi ---")
        model = nv.models.resnet18(num_classes=10)

        stats = nv.quantize_model(model)

        print(f"  [OK] Kuantize Edilen Parametre Sayisi: {stats['total_quantized_params']:,}")
        print(f"  [OK] FP32 Agirlik Bellek Boyutu    : {stats['fp32_memory_kb']:.2f} KB")
        print(f"  [OK] INT8 Agirlik Bellek Boyutu    : {stats['int8_memory_kb']:.2f} KB")
        print(f"  [OK] Bellek Tasarrufu Orani (%)   : %{stats['memory_reduction_percent']:.2f}")

        self.assertGreater(stats['memory_reduction_percent'], 70.0)


if __name__ == "__main__":
    print("==========================================")
    print("newton-vision v2.0: AVX2 SIMD & INT8 Quantization Testleri")
    print("==========================================")
    unittest.main()
