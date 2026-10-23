"""benchmarks/bench_vs_pytorch.py - Scientific & Statistical Benchmark Suite: newton-vision vs PyTorch 2.6 CPU."""

import time
import sys
import os
import platform
import numpy as np

import newton_vision as nv

try:
    import torch
    import torchvision.models as torch_models
    from torchvision.models.vision_transformer import VisionTransformer as TorchVisionTransformer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

WARMUP = 3
RUNS = 10


def get_cpu_info():
    """Returns processor brand and environment details."""
    cpu_name = platform.processor() or platform.machine()
    num_threads = os.environ.get("OMP_NUM_THREADS", str(os.cpu_count() or 1))
    return cpu_name, num_threads


def compute_statistics(latencies_ms, batch_size):
    """Computes mean, std, min, max, and FPS from latencies list."""
    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    fps = float((batch_size * 1000.0) / mean_ms) if mean_ms > 0 else 0.0
    return {
        "mean": mean_ms,
        "std": std_ms,
        "min": min_ms,
        "max": max_ms,
        "fps": fps
    }


def benchmark_model_nv(model, dummy_data, warmup=WARMUP, runs=RUNS):
    """Benchmarks a newton-vision model with statistical sampling (eval + no_grad, PyTorch tarafıyla aynı)."""
    x = nv.Tensor(dummy_data, requires_grad=False)
    model.eval()

    with nv.no_grad():
        # Warmup iterations
        for _ in range(warmup):
            _ = model(x)

        # Recorded timing runs
        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return compute_statistics(latencies, dummy_data.shape[0])


def torch_vit_tiny():
    """nv.models.vit_tiny ile aynı boyutlarda torchvision ViT (embed=192, depth=12, heads=3, mlp=768)."""
    return TorchVisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=3,
                                  hidden_dim=192, mlp_dim=768, num_classes=1000)


def count_params_nv(model):
    return int(sum(p.data.size for p in model.parameters()))


def count_params_torch(model):
    return int(sum(p.numel() for p in model.parameters()))


def benchmark_model_torch(model, dummy_data, warmup=WARMUP, runs=RUNS):
    """Benchmarks a PyTorch model with statistical sampling."""
    if not HAS_TORCH:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "fps": 0.0}

    dummy_tensor = torch.from_numpy(dummy_data)
    model = model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_tensor)

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy_tensor)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return compute_statistics(latencies, dummy_data.shape[0])


def run_scientific_benchmarks():
    cpu_name, num_threads = get_cpu_info()

    print("=======================================================================")
    print(f"🌌 newton-vision v{nv.__version__} vs PyTorch CPU — Bilimsel Benchmark Raporu")
    print("=======================================================================")
    print(f"• CPU Model          : {cpu_name}")
    print(f"• OpenMP Thread Sayısı: {num_threads} Threads")
    print(f"• Python Sürümü      : {sys.version.split()[0]}")
    print(f"• PyTorch Sürümü     : {torch.__version__ if HAS_TORCH else 'Yüklü Değil'}")
    print(f"• AVX2 Hardware      : {nv._core.has_avx2()}")
    print(f"• Veri Hassasiyeti    : FP32 (Single Precision)")
    print(f"• İstatistiki Örneklem : Warmup={WARMUP}, Runs={RUNS} Iterations (eval + no_grad)")
    print("=======================================================================\n")

    data_224 = np.random.randn(1, 3, 224, 224).astype(np.float32)

    benchmarks = [
        ("ResNet-18", lambda: nv.models.resnet18(num_classes=1000), lambda: torch_models.resnet18()),
        ("MobileNetV2", lambda: nv.models.mobilenet_v2(num_classes=1000), lambda: torch_models.mobilenet_v2()),
        ("ViT-Tiny (Transformer)", lambda: nv.models.vit_tiny(num_classes=1000), torch_vit_tiny),
    ]

    report_lines = []
    report_lines.append(f"# 📊 newton-vision v{nv.__version__} vs PyTorch CPU Bilimsel Karşılaştırma Raporu\n")
    report_lines.append(f"**Test Çevresi:** CPU: `{cpu_name}` | Threads: `{num_threads}` | Precision: `FP32` | Iterations: `Warmup={WARMUP}, Runs={RUNS}` | Mod: `eval + no_grad`\n")
    report_lines.append("| Model | Framework | Parametre | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | FPS | Karşılaştırma |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|")

    for name, nv_fn, torch_fn in benchmarks:
        print(f"▶ Scientific Sampling: {name}...")
        nv_model = nv_fn()
        nv_params = count_params_nv(nv_model)
        nv_stats = benchmark_model_nv(nv_model, data_224)

        if HAS_TORCH:
            t_model = torch_fn()
            t_params = count_params_torch(t_model)
            if t_params != nv_params:
                # Farklı boyuttaki modelleri karşılaştırmak geçersiz sonuç üretir (eski ViT-B/16 hatası)
                raise RuntimeError(f"{name}: parametre sayıları uyuşmuyor (newton-vision {nv_params:,} vs PyTorch {t_params:,})")
            t_stats = benchmark_model_torch(t_model, data_224)
            ratio = t_stats["mean"] / nv_stats["mean"] if nv_stats["mean"] > 0 else 1.0
            ratio_str = f"{ratio:.2f}x Hızlı" if ratio >= 1.0 else f"{1.0/ratio:.2f}x Yavaş"
        else:
            t_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "fps": 0.0}
            ratio_str = "N/A"

        print(f"  • newton-vision : {nv_stats['mean']:.2f} ms ± {nv_stats['std']:.2f} ms (Min: {nv_stats['min']:.2f}, Max: {nv_stats['max']:.2f}) | {nv_stats['fps']:.1f} FPS")
        if HAS_TORCH:
            print(f"  • PyTorch CPU   : {t_stats['mean']:.2f} ms ± {t_stats['std']:.2f} ms (Min: {t_stats['min']:.2f}, Max: {t_stats['max']:.2f}) | {t_stats['fps']:.1f} FPS")
            print(f"  • Oran           : {ratio_str}\n")

        report_lines.append(f"| **{name}** | newton-vision | {nv_params:,} | **{nv_stats['mean']:.2f}** | ±{nv_stats['std']:.2f} | {nv_stats['min']:.2f} | {nv_stats['max']:.2f} | **{nv_stats['fps']:.1f}** | **{ratio_str}** |")
        if HAS_TORCH:
            report_lines.append(f"| | PyTorch CPU | {t_params:,} | {t_stats['mean']:.2f} | ±{t_stats['std']:.2f} | {t_stats['min']:.2f} | {t_stats['max']:.2f} | {t_stats['fps']:.1f} | Base |")

    markdown_report = "\n".join(report_lines)
    print("=======================================================================")
    print(markdown_report)
    print("=======================================================================")

    # Elle derlenen BENCHMARK_RESULTS.md'nin üzerine yazmamak için ayrı dosya
    report_path = os.path.join(os.path.dirname(__file__), "BENCH_VS_PYTORCH.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_report + "\n")
    print(f"\n✅ Bilimsel Benchmark raporu başarıyla güncellendi: {report_path}")


if __name__ == "__main__":
    run_scientific_benchmarks()
