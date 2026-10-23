"""newton_vision.principia - Sistem & Cihaz Yönetimi (CPU / CUDA GPU)."""

from __future__ import annotations
import newton_vision._core as _core

_CURRENT_DEVICE = "cpu"


def is_cuda_available() -> bool:
    """CUDA GPU ivmelendirme cihazı mevcut mu?"""
    return _core.has_cuda()


def set_device(device: str) -> None:
    """Çalışma cihazını ayarla ('cpu' veya 'cuda')."""
    global _CURRENT_DEVICE
    dev = device.lower()
    if dev == "cuda":
        if not is_cuda_available():
            raise RuntimeError("CUDA GPU cihazı sistemde mevcut veya derlenmiş değil!")
        _CURRENT_DEVICE = "cuda"
    elif dev == "cpu":
        _CURRENT_DEVICE = "cpu"
    else:
        raise ValueError(f"Bilinmeyen cihaz: {device}. Beklenen 'cpu' veya 'cuda'")


def get_device() -> str:
    """Mevcut cihaz adını döndürür."""
    return _CURRENT_DEVICE
