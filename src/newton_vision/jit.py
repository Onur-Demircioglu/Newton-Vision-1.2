"""newton_vision.jit - Model Serileştirme (.nv & .nv.bin), ONNX Export ve Canlı Çıkarım Motoru (InferenceEngine)."""

from __future__ import annotations
import os
import time
import struct
import numpy as np
import newton_vision as nv


def extract_state_dict(model) -> dict[str, np.ndarray]:
    """Modelin parametre ve tamponlarını state_dict sözlüğü olarak çıkarır."""
    if hasattr(model, "state_dict") and callable(model.state_dict):
        sd = model.state_dict()
        if sd:
            return sd

    state_dict = {}

    def _traverse(obj, prefix=""):
        if hasattr(obj, "w") and isinstance(obj.w, nv.Tensor):
            state_dict[f"{prefix}weight"] = obj.w.data.copy()
        if hasattr(obj, "bias") and obj.bias is not None and isinstance(obj.bias, nv.Tensor):
            state_dict[f"{prefix}bias"] = obj.bias.data.copy()
        if hasattr(obj, "gamma") and isinstance(obj.gamma, nv.Tensor):
            state_dict[f"{prefix}weight"] = obj.gamma.data.copy()
        if hasattr(obj, "beta") and isinstance(obj.beta, nv.Tensor):
            state_dict[f"{prefix}bias"] = obj.beta.data.copy()
        if hasattr(obj, "running_mean"):
            state_dict[f"{prefix}running_mean"] = np.asarray(obj.running_mean, dtype=np.float32).copy()
        if hasattr(obj, "running_var"):
            state_dict[f"{prefix}running_var"] = np.asarray(obj.running_var, dtype=np.float32).copy()

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (list, tuple)):
                    for idx, elem in enumerate(v):
                        if hasattr(elem, "__dict__"):
                            _traverse(elem, f"{prefix}{k}.{idx}.")
                elif hasattr(v, "__dict__") and not isinstance(v, (nv.Tensor, np.ndarray)):
                    _traverse(v, f"{prefix}{k}.")

    _traverse(model)
    return state_dict


def save_model(model, filepath: str) -> str:
    """Model ağırlıklarını sıkıştırılmış ikili .nv dosya formatında kaydeder."""
    if not filepath.endswith(".nv"):
        filepath += ".nv"

    state_dict = extract_state_dict(model)
    with open(filepath, "wb") as f:
        np.savez_compressed(f, **state_dict)

    print(f"  [OK] Model ikili formatta diske kaydedildi: {filepath}")
    return filepath


def load_model(model, filepath: str):
    """Sıkıştırılmış ikili .nv dosyasından model ağırlıklarını yükler."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")

    # Guvenli serilestirme: RCE (Remote Code Execution) saldirilarini onlemek icin pickle devre disi
    data = np.load(filepath, allow_pickle=False)
    state_dict = {k: data[k] for k in data.files}
    # Kendi checkpoint'imiz: her anahtar eşleşmeli, aksi halde yarım yüklenmiş model döner
    nv.load_state_dict(model, state_dict, strict=True)
    print(f"  [OK] Model ikili formatta diske yuklendi: {filepath}")
    return model


def save_model_bin(model, filepath: str) -> str:
    """Model ağırlıklarını saf C++ ikili (.nv.bin) formatında kaydeder."""
    if not filepath.endswith(".nv.bin"):
        filepath += ".nv.bin"

    state_dict = extract_state_dict(model)
    with open(filepath, "wb") as f:
        f.write(b"NVBIN100")
        f.write(struct.pack("<I", len(state_dict)))

        for name, data in state_dict.items():
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            shape = data.shape
            if len(shape) == 1:
                dims = (1, int(shape[0]), 1, 1)
            elif len(shape) == 2:
                dims = (int(shape[0]), int(shape[1]), 1, 1)
            elif len(shape) == 4:
                dims = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
            else:
                dims = (1, int(np.prod(shape)), 1, 1)

            f.write(struct.pack("<4i", *dims))
            float_data = np.ascontiguousarray(data, dtype=np.float32)
            f.write(struct.pack("<Q", float_data.size))
            f.write(float_data.tobytes())

    print(f"  [OK] Model saf C++ ikili formatta (.nv.bin) diske kaydedildi: {filepath}")
    return filepath


def load_model_bin(model, filepath: str):
    """Saf C++ ikili (.nv.bin) dosyasından model ağırlıklarını yükler."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {filepath}")

    state_dict = {}
    with open(filepath, "rb") as f:
        magic = f.read(8)
        if magic != b"NVBIN100":
            raise ValueError(f"Geçersiz C++ ikili model formatı: {filepath}")

        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            n, c, h, w = struct.unpack("<4i", f.read(16))
            (float_count,) = struct.unpack("<Q", f.read(8))
            float_bytes = f.read(float_count * 4)
            arr = np.frombuffer(float_bytes, dtype=np.float32).reshape(n, c, h, w)
            state_dict[name] = arr

    nv.load_state_dict(model, state_dict, strict=True)
    print(f"  [OK] Model saf C++ ikili formatindan (.nv.bin) diske yuklendi: {filepath}")
    return model


def export_onnx(model, dummy_input: np.ndarray | nv.Tensor, filepath: str, verify: bool = True) -> str:
    """Modeli gerçek bir ONNX (protobuf, opset 17) dosyasına dışa aktarır.

    Model `dummy_input` ile bir kez izlenir; graf bu girdinin şekline sabittir (batch dahil).
    verify=True ve onnxruntime kuruluysa çıktı, iki farklı girdiyle newton-vision'a karşı
    doğrulanır; uyuşmazlıkta RuntimeError. Gereksinim: `pip install newton-vision[onnx]`.
    Ayrıntılar ve sınırlar: `newton_vision.onnx_export`.
    """
    from newton_vision.onnx_export import export

    if not filepath.endswith(".onnx"):
        filepath += ".onnx"
    export(model, dummy_input, filepath, verify=verify)
    print(f"  [OK] Model ONNX (opset 17) formatina aktarildi: {filepath}")
    return filepath


class InferenceEngine:
    """Yüksek hızlı canlı çıkarım (inference) motoru & FPS / Gecikme takipçisi."""
    def __init__(self, model):
        self.model = model
        self.latency_history = []

    def predict(self, input_frame: np.ndarray | nv.Tensor) -> tuple[nv.Tensor, float, float]:
        """Kare üzerinde çıkarım yapar, gecikme (ms) ve FPS değerlerini döndürür."""
        start_time = time.perf_counter()

        output = self.model(input_frame)

        latency_sec = time.perf_counter() - start_time
        latency_ms = latency_sec * 1000.0
        fps = 1.0 / latency_sec if latency_sec > 0 else 0.0

        self.latency_history.append(latency_ms)
        return output, latency_ms, fps

    def get_average_stats(self) -> tuple[float, float]:
        """Ortalama gecikme (ms) ve ortalama FPS değerlerini döndürür."""
        if not self.latency_history:
            return 0.0, 0.0
        avg_latency_ms = float(np.mean(self.latency_history))
        avg_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
        return avg_latency_ms, avg_fps
