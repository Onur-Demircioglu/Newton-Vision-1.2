"""Model Serialization, ONNX Export & Inference Engine Testi (tests/test_jit.py).

.nv ikili formatta model kaydetme/yükleme, ONNX grafik çıktısı ve InferenceEngine FPS ölçümünü test eder.
"""

from __future__ import annotations
import os
import unittest
import numpy as np
import newton_vision as nv

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_serialization():
    print("--- Model Serileştirme (.nv İkili Checkpoint) Testi ---")
    conv = nv.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)

    # Set arbitrary weights
    conv.w.data.fill(3.14)

    ckpt_path = os.path.join(TEST_DIR, "test_model.nv")
    nv.save_model(conv, ckpt_path)
    assert os.path.exists(ckpt_path), "Model .nv dosyası kaydedilemedi!"

    # Create new instance and restore
    new_conv = nv.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    nv.load_model(new_conv, ckpt_path)

    assert np.allclose(new_conv.w.data, 3.14), "Yüklenen ağırlıklar uyuşmuyor!"
    print("  ✓ Model .nv formatında başarıyla kaydedildi ve yüklendi")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def test_onnx_exporter():
    print("--- ONNX Export Testi ---")
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("onnx kurulu değil (pip install newton-vision[onnx])")
    model = nv.models.resnet18(num_classes=10)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    onnx_path = os.path.join(TEST_DIR, "resnet18.onnx")
    nv.export_onnx(model, dummy_input, onnx_path)
    assert os.path.exists(onnx_path), "ONNX dosyası üretilemedi!"
    print("  ✓ ResNet-18 ONNX grafik çıktısı oluşturuldu")

    if os.path.exists(onnx_path):
        os.remove(onnx_path)


def test_inference_engine():
    print("--- InferenceEngine Canlı Çıkarım & FPS Sayaç Testi ---")
    model = nv.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    engine = nv.InferenceEngine(model)

    for _ in range(5):
        frame = np.random.randn(1, 1, 64, 64).astype(np.float32)
        out, lat_ms, fps = engine.predict(frame)
        assert out.shape == (1, 4, 64, 64), f"Çıkarım boyutu yanlış: {out.shape}"

    avg_ms, avg_fps = engine.get_average_stats()
    print(f"  ✓ Ortalama Çıkarım Gecikmesi: {avg_ms:.2f} ms | Hız: {avg_fps:.1f} FPS")


def main():
    print("==========================================")
    print("newton-vision v0.6: JIT & InferenceEngine Testleri")
    print("==========================================")
    test_serialization()
    test_onnx_exporter()
    test_inference_engine()
    print("\n✅ TÜM JIT VE SERİLEŞTİRME TESTLERİ BAŞARIYLA GEÇTİ!")


def load_tests(loader, tests, pattern):
    """`python -m unittest discover` modül düzeyindeki test_* fonksiyonlarını da çalıştırsın diye."""
    funcs = [fn for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in funcs)


if __name__ == "__main__":
    main()
