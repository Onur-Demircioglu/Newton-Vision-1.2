"""newton_vision.transforms - Örnek başına veri artırma ve ön işleme dönüşümleri.

Dönüşümler tek bir örnek (C, H, W) NumPy dizisi üzerinde çalışır ve yeni bir dizi döndürür
(girdiyi değiştirmez). Rastgelelik `np.random` ile üretilir; `np.random.seed` ile tekrarlanabilir.
Kullanım: `nv.TensorDataset(x, y, transform=nv.transforms.Compose([...]))`.
"""

from __future__ import annotations
from typing import Callable, Sequence
import numpy as np


class Compose:
    """Dönüşümleri sırayla uygular."""
    def __init__(self, transforms: Sequence[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


def _check_chw(name: str, x: np.ndarray):
    if x.ndim != 3:
        raise ValueError(f"{name} (C, H, W) şeklinde tek örnek bekler, alınan: {x.shape}")


class RandomHorizontalFlip:
    """p olasılıkla görüntüyü yatay çevirir (genişlik ekseni)."""
    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p [0, 1] aralığında olmalı, alınan: {p}")
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _check_chw("RandomHorizontalFlip", x)
        if np.random.rand() < self.p:
            return np.ascontiguousarray(x[:, :, ::-1])
        return x.copy()


class RandomCrop:
    """Önce isteğe bağlı sıfır dolgu ekler, sonra rastgele bir size x size pencere kırpar."""
    def __init__(self, size: int | tuple[int, int], padding: int = 0):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if padding < 0:
            raise ValueError(f"padding >= 0 olmalı, alınan: {padding}")
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _check_chw("RandomCrop", x)
        p = self.padding
        if p > 0:
            x = np.pad(x, ((0, 0), (p, p), (p, p)))
        th, tw = self.size
        _, h, w = x.shape
        if th > h or tw > w:
            raise ValueError(f"RandomCrop: kırpma boyutu {self.size}, (dolgulu) girdi {h}x{w}'den büyük")
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        return np.ascontiguousarray(x[:, i:i + th, j:j + tw])


class Normalize:
    """Kanal bazında (x - mean) / std uygular."""
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)
        if np.any(self.std == 0):
            raise ValueError("Normalize: std sıfır olamaz")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        _check_chw("Normalize", x)
        if self.mean.shape[0] not in (1, x.shape[0]):
            raise ValueError(f"Normalize: {self.mean.shape[0]} kanal için mean verildi, girdi {x.shape[0]} kanallı")
        return ((x - self.mean) / self.std).astype(np.float32)


__all__ = ["Compose", "RandomHorizontalFlip", "RandomCrop", "Normalize"]
