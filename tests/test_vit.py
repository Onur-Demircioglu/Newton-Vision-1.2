"""newton-vision v2.0: Vision Transformer (ViT), MHSA, LayerNorm ve GELU Testleri."""

import unittest
import numpy as np
import newton_vision as nv


class TestVisionTransformer(unittest.TestCase):

    def test_gelu(self):
        print("\n--- GELU Aktivasyon Testi ---")
        x_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
        x = nv.Tensor(x_data, requires_grad=True)
        gelu = nv.optics.GELU()
        y = gelu(x)

        # Reference formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        ref_y = 0.5 * x_data * (1.0 + np.tanh(sqrt_2_pi * (x_data + 0.044715 * (x_data ** 3))))

        diff = float(np.max(np.abs(y.data - ref_y)))
        print(f"  [OK] GELU C++ Output vs Referans Maksimum Fark: {diff:.6f}")
        self.assertLess(diff, 1e-4)

    def test_layernorm(self):
        print("\n--- LayerNorm (Katman Normalizasyonu) Testi ---")
        x_data = np.random.randn(4, 16, 64).astype(np.float32) * 5.0 + 10.0
        x = nv.Tensor(x_data, requires_grad=True)
        ln = nv.optics.LayerNorm(64)
        y = ln(x)

        # Verify normalized mean is close to 0 and std is close to 1 across feature_dim (axis=-1)
        mean_after = float(np.max(np.abs(np.mean(y.data, axis=-1))))
        std_after = float(np.max(np.abs(np.std(y.data, axis=-1) - 1.0)))

        print(f"  [OK] LayerNorm Sonrasi Maksimum Ortalama Saptmasi: {mean_after:.6f}")
        print(f"  [OK] LayerNorm Sonrasi Maksimum Standart Sapma Saptmasi: {std_after:.6f}")
        self.assertLess(mean_after, 1e-3)
        self.assertLess(std_after, 1e-2)

    def test_multi_head_attention(self):
        print("\n--- Multi-Head Self-Attention (MHSA) Testi ---")
        # Input shape: (Batch=2, Seq_Len=16, Embed_Dim=64)
        x_data = np.random.randn(2, 16, 64).astype(np.float32)
        x = nv.Tensor(x_data, requires_grad=True)
        
        mhsa = nv.optics.MultiHeadAttention(embed_dim=64, num_heads=4)
        out = mhsa(x)

        self.assertEqual(out.data.shape, (2, 16, 64))
        print(f"  [OK] MHSA Forward Pass Cikti Boyutu: {out.data.shape}")

        # Backward Pass Test with random gradient array to prevent zero grads
        grad_mask = np.random.randn(*out.data.shape).astype(np.float32)
        out.backward(grad_mask)
        self.assertIsNotNone(x.grad)
        self.assertGreater(float(np.max(np.abs(x.grad))), 0.0)
        self.assertGreater(float(np.max(np.abs(mhsa.q_proj.w.grad))), 0.0)
        self.assertGreater(float(np.max(np.abs(mhsa.k_proj.w.grad))), 0.0)
        self.assertGreater(float(np.max(np.abs(mhsa.v_proj.w.grad))), 0.0)
        print(f"  [OK] MHSA Autograd Backward Pass Gradyan Boyutu: {x.grad.shape}")

    def test_layernorm_backward_parameters(self):
        x_data = np.random.randn(2, 4, 8).astype(np.float32)
        x = nv.Tensor(x_data, requires_grad=True)
        ln = nv.optics.LayerNorm(8)

        # Rastgele agirliklandirilmis toplam kullanarak gradyanlarin sifirlanmasini onluyoruz
        grad_mask = np.random.randn(2, 4, 8).astype(np.float32)
        loss = (ln(x) * grad_mask).sum()
        loss.backward()

        self.assertGreater(float(np.max(np.abs(x.grad))), 0.0)
        self.assertGreater(float(np.max(np.abs(ln.gamma.grad))), 0.0)
        self.assertEqual(ln.beta.grad.shape, (8,))

    def test_vit_tiny_forward(self):
        print("\n--- ViT-Tiny Vision Transformer Model Forward Pass Testi ---")
        # Dummy Image input: (Batch=1, Channels=3, Height=224, Width=224)
        img_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        img = nv.Tensor(img_data, requires_grad=False)

        # ViT-Tiny classifier (10 classes)
        vit = nv.models.vit_tiny(num_classes=10, img_size=224, patch_size=16)
        logits = vit(img)

        self.assertEqual(logits.data.shape, (1, 10))
        predicted_class = int(np.argmax(logits.data))
        print(f"  [OK] ViT-Tiny Logits Cikti Boyutu: {logits.data.shape}")
        print(f"  [OK] En Yuksek Tahmin Edilen Sinif: {predicted_class}")


if __name__ == "__main__":
    print("==========================================")
    print("newton-vision v2.0: Vision Transformer (ViT) & Attention Testleri")
    print("==========================================")
    unittest.main()
