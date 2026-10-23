"""Gerçek Zamanlı Görsel Akışı & FPS Performans Demosu (examples/realtime_stream_demo.py).

newton-vision InferenceEngine ile kare kare canlı görüntü akışını işler,
milisaniyelik gecikmeleri (latency) ve saniye başına kare sayısını (FPS) raporlar.
"""

from __future__ import annotations
import time
import numpy as np
import newton_vision as nv


def main():
    print("==================================================")
    print("newton-vision: GERÇEK ZAMANLI CANLI ÇIKARIM SUNUCUSU")
    print("==================================================")
    print("🤖 Model Yükleniyor: LeNet-Style Multi-Layer CNN...")

    class StreamModel:
        def __init__(self):
            self.conv1 = nv.Conv2d(3, 16, 3, padding=1)
            self.relu1 = nv.ReLU()
            self.pool1 = nv.MaxPool2d(2)
            self.conv2 = nv.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nv.ReLU()
            self.pool2 = nv.MaxPool2d(2)
            self.fc = nv.Linear(32 * 16 * 16, 10)

        def __call__(self, x):
            out = self.pool1(self.relu1(self.conv1(x)))
            out = self.pool2(self.relu2(self.conv2(out)))
            N = out.data.shape[0]
            flat = out.data.reshape(N, -1)
            return self.fc(flat)

    model = StreamModel()
    engine = nv.InferenceEngine(model)

    print("🎥 10 Kare Canlı Video Akışı Başlatılıyor (64x64 RGB Görseller)...\n")

    for frame_idx in range(1, 11):
        # 64x64 RGB canlı kamera karesi simülasyonu
        frame_data = np.random.randn(1, 3, 64, 64).astype(np.float32)

        out, latency_ms, fps = engine.predict(frame_data)
        print(f"  Frame [{frame_idx:2d}/10] | Gecikme: {latency_ms:6.2f} ms | Anlık Performans: {fps:6.1f} FPS")
        time.sleep(0.01)

    avg_ms, avg_fps = engine.get_average_stats()
    print("\n--------------------------------------------------")
    print(f"📊 TOPLAM PERFORMANS ÖZETİ:")
    print(f"  • Ortalama Kare Gecikmesi (Latency) : {avg_ms:.2f} ms")
    print(f"  • Ortalama Çıkarım Hızı (FPS)        : {avg_fps:.1f} FPS")
    print("--------------------------------------------------")

    # Save model checkpoint
    ckpt_file = nv.save_model(model, "stream_model.nv")
    print("✅ CANLI ÇIKARIM SUNUCUSU DEMOSU BAŞARIYLA TAMAMLANDI!")


if __name__ == "__main__":
    main()
