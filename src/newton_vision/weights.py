"""PyTorch Ağırlık Dönüştürücüsü & State Dict Yükleyici (newton_vision.weights)."""

from __future__ import annotations
import warnings
from typing import NamedTuple
import numpy as np
from newton_vision.module import Module
from newton_vision.tensor import Tensor


# PyTorch'a özgü, newton-vision'da karşılığı olmayan ve bilinçli olarak yok sayılan tamponlar
_IGNORED_SUFFIXES = ("num_batches_tracked",)


class LoadResult(NamedTuple):
    """`load_state_dict` sonucu (PyTorch `_IncompatibleKeys` ile aynı alan adları)."""
    missing_keys: list[str]
    unexpected_keys: list[str]


def _module_prefixes(model: Module) -> dict[int, str]:
    """Her alt modülün state_dict anahtar önekini (ör. 'layer1.0.') çıkarır."""
    prefixes: dict[int, str] = {}

    def _walk(module: Module, prefix: str):
        prefixes.setdefault(id(module), prefix)
        for name, child in module._modules.items():
            _walk(child, f"{prefix}{name}.")

    _walk(model, "")
    return prefixes


def _resolve_module(model: Module, segments: list[str]) -> Module | None:
    """Anahtarın modül yolunu çözer.

    newton-vision adlarına ek olarak iki PyTorch biçimini de eşler:
      * 'blocks.0'     -> 'blocks_0'   (Python listesi olarak atanan alt modüller)
      * 'downsample.0' -> kayıt sırasına göre 0. alt modül (torch.nn.Sequential indeksleri)
    """
    curr = model
    i = 0
    while i < len(segments):
        seg = segments[i]
        children = curr._modules
        if seg in children:
            curr = children[seg]
            i += 1
            continue
        if i + 1 < len(segments) and segments[i + 1].isdigit() and f"{seg}_{segments[i + 1]}" in children:
            curr = children[f"{seg}_{segments[i + 1]}"]
            i += 2
            continue
        if seg.isdigit() and int(seg) < len(children):
            curr = list(children.values())[int(seg)]
            i += 1
            continue
        return None
    return curr


def _resolve_attr(module: Module, attr: str) -> str | None:
    """Parametre/tampon adını çözer (PyTorch 'weight'/'bias' -> 'w'/'gamma'/'beta')."""
    if attr in module._parameters or attr in module._buffers:
        return attr
    if attr == "weight":
        for native in ("w", "gamma"):
            if native in module._parameters:
                return native
    if attr == "bias" and "beta" in module._parameters:
        return "beta"
    return None


def _convert(key: str, value: np.ndarray, target_shape: tuple[int, ...], native_attr: str, source_attr: str) -> np.ndarray:
    """Gelen diziyi hedef şekle uyarlar; uyarlanamıyorsa ValueError fırlatır."""
    arr = np.asarray(value, dtype=np.float32)

    if native_attr == "w" and len(target_shape) == 2:
        # .nv.bin 2D ağırlıkları (a, b, 1, 1) olarak saklar
        if arr.ndim == 4 and arr.shape[2:] == (1, 1):
            arr = arr.reshape(arr.shape[:2])
        if arr.ndim == 2:
            is_square = target_shape[0] == target_shape[1]
            if arr.shape == target_shape[::-1] and not is_square:
                # PyTorch Linear (out, in) -> newton-vision (in, out)
                return arr.T.copy()
            if arr.shape == target_shape and is_square and source_attr == "weight":
                # Kare matriste şekilden yön anlaşılamaz: 'weight' adı PyTorch/C++ (out, in) düzenidir
                return arr.T.copy()

    if arr.shape != target_shape:
        squeezed_src = [d for d in arr.shape if d != 1]
        squeezed_dst = [d for d in target_shape if d != 1]
        is_bin_flattened = arr.shape == (1, arr.size, 1, 1) and arr.size == int(np.prod(target_shape))
        if squeezed_src == squeezed_dst or is_bin_flattened:
            # .nv.bin 1D tensörleri (1, n, 1, 1), 3D/5D+ tensörleri C-sırasında (1, N, 1, 1) olarak saklar
            arr = arr.reshape(target_shape)
        else:
            raise ValueError(
                f"[load_state_dict] '{key}' şekli uyuşmuyor: gelen {arr.shape}, beklenen {target_shape}"
            )
    return arr.copy()


def load_state_dict(model: Module, state_dict: dict[str, np.ndarray], strict: bool = False) -> LoadResult:
    """state_dict sözlüğünü (newton-vision veya PyTorch adlandırması) modele aktarır.

    * PyTorch Linear (out_features, in_features) ağırlıkları (in_features, out_features)
      düzenine otomatik çevrilir. Kare matrislerde yön adla belirlenir: 'weight' = PyTorch
      düzeni (transpoz edilir), 'w' = newton-vision düzeni (olduğu gibi kopyalanır).
    * Şekli uyarlanamayan her anahtar ValueError fırlatır; hiçbir değer yazılmadan önce
      tüm anahtarlar doğrulanır, yani hata durumunda model değişmeden kalır.
    * strict=True: eksik veya beklenmeyen anahtar varsa RuntimeError.
      strict=False: bu anahtarlar için uyarı (UserWarning) verilir ve LoadResult döner.
    * 'num_batches_tracked' gibi PyTorch'a özgü tamponlar bilinçli olarak yok sayılır.
    """
    if not isinstance(model, Module):
        raise TypeError(f"load_state_dict bir Module bekler, alınan: {type(model)}")

    prefixes = _module_prefixes(model)
    expected_keys = list(model.state_dict().keys())

    planned: list[tuple[Module, str, np.ndarray]] = []
    loaded_keys: set[str] = set()
    unexpected: list[str] = []

    for key, value in state_dict.items():
        if key.split(".")[-1] in _IGNORED_SUFFIXES:
            continue

        parts = key.split(".")
        module = _resolve_module(model, parts[:-1])
        native_attr = _resolve_attr(module, parts[-1]) if module is not None else None
        if native_attr is None:
            unexpected.append(key)
            continue

        native_key = f"{prefixes[id(module)]}{native_attr}"
        if native_key in loaded_keys:
            raise ValueError(f"[load_state_dict] '{key}' anahtarı '{native_key}' hedefine ikinci kez eşlendi")

        current = module._parameters.get(native_attr)
        if current is None:
            current = module._buffers[native_attr]
        target_shape = tuple((current.data if isinstance(current, Tensor) else np.asarray(current)).shape)

        arr = _convert(key, value, target_shape, native_attr, parts[-1])
        planned.append((module, native_attr, arr))
        loaded_keys.add(native_key)

    missing = [k for k in expected_keys if k not in loaded_keys]

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"[load_state_dict] strict=True: eksik anahtarlar {missing}, beklenmeyen anahtarlar {unexpected}"
        )

    for module, native_attr, arr in planned:
        if native_attr in module._parameters:
            module._parameters[native_attr].data = arr
        else:
            setattr(module, native_attr, arr)

    if missing or unexpected:
        warnings.warn(
            f"[load_state_dict] {len(missing)} eksik anahtar (ilk 5: {missing[:5]}), "
            f"{len(unexpected)} beklenmeyen anahtar (ilk 5: {unexpected[:5]}) — bu değerler yüklenmedi.",
            stacklevel=2,
        )

    return LoadResult(missing_keys=missing, unexpected_keys=unexpected)
