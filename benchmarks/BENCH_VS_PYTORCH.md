# 📊 newton-vision v1.1.6 vs PyTorch CPU Bilimsel Karşılaştırma Raporu

**Test Çevresi:** CPU: `Intel64 Family 6 Model 141 Stepping 1, GenuineIntel` | Threads: `12` | Precision: `FP32` | Iterations: `Warmup=3, Runs=10` | Mod: `eval + no_grad`

| Model | Framework | Parametre | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | FPS | Karşılaştırma |
|---|---|---|---|---|---|---|---|---|
| **ResNet-18** | newton-vision | 11,689,512 | **122.97** | ±6.09 | 112.91 | 131.35 | **8.1** | **4.19x Yavaş** |
| | PyTorch CPU | 11,689,512 | 29.34 | ±1.39 | 27.42 | 31.80 | 34.1 | Base |
| **MobileNetV2** | newton-vision | 3,504,872 | **109.18** | ±12.57 | 91.22 | 128.67 | **9.2** | **4.34x Yavaş** |
| | PyTorch CPU | 3,504,872 | 25.18 | ±6.76 | 19.78 | 43.84 | 39.7 | Base |
| **ViT-Tiny (Transformer)** | newton-vision | 5,717,416 | **61.25** | ±7.03 | 50.52 | 76.76 | **16.3** | **2.53x Yavaş** |
| | PyTorch CPU | 5,717,416 | 24.18 | ±1.21 | 22.14 | 26.30 | 41.3 | Base |
