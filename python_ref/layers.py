"""Saf Python referans katmanlar — C++ çekirdeğinin altın standardı.

Bağımsız bir yoldan (NumPy vectorize) doğru sonucu üretir. C++ (pybind11)
çıktısı buradaki çıktıya eşit olmak zorunda. Eşit değilse C++'ta bug vardır.
"""

from __future__ import annotations

import numpy as np


def _check_float32(name: str, a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        raise TypeError(f"{name} beklenen dtype float32, gelen {a.dtype}")
    return a


def conv2d(
    x: np.ndarray,
    w: np.ndarray,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """2D cross-correlation (CNN conv).

    x: (N, C_in, H, W); w: (C_out, C_in, kH, kW); bias: (C_out,)
    -> (N, C_out, H_out, W_out)   H_out=(H+2p-kH)//s+1
    """
    x = _check_float32("x", x)
    w = _check_float32("w", w)
    if bias is not None:
        bias = _check_float32("bias", bias)

    N, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = w.shape
    if C_in != C_in_w:
        raise ValueError(f"kanal uyumsuz: x {C_in} vs w {C_in_w}")

    if padding > 0:
        xp = np.zeros((N, C_in, H + 2 * padding, W + 2 * padding), dtype=np.float32)
        xp[:, :, padding : padding + H, padding : padding + W] = x
        x = xp
        H, W = x.shape[2], x.shape[3]

    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"geçersiz çıktı boyutu: H_out={H_out} W_out={W_out}")

    s = stride
    # im2col: her (kh, kw) dilimini cols'a yazar.
    # cols: (N, C_in, kH, kW, H_out, W_out) -> (N, H_out*W_out, C_in*kH*kW)
    cols = np.empty((N, C_in, kH, kW, H_out, W_out), dtype=np.float32)
    for kh in range(kH):
        for kw in range(kW):
            patch = x[:, :, kh : kh + s * H_out : s, kw : kw + s * W_out : s]
            cols[:, :, kh, kw, :, :] = patch  # (N, C_in, H_out, W_out)

    cols = cols.reshape(N, C_in * kH * kW, H_out * W_out)
    # w: (C_out, C_in*kH*kW) @ (N, C_in*kH*kW, L) -> (N, C_out, L)
    w2d = w.reshape(C_out, C_in * kH * kW)
    y = np.einsum("oc,ncl->nol", w2d, cols, optimize=True)  # (N, C_out, L)
    y = y.reshape(N, C_out, H_out, W_out)

    if bias is not None:
        y += bias.reshape(1, C_out, 1, 1)
    return y


def relu(x: np.ndarray) -> np.ndarray:
    """max(0, x)."""
    x = _check_float32("x", x)
    return np.maximum(x, np.float32(0))


def maxpool2d(x: np.ndarray, kernel_size: int, stride: int | None = None) -> np.ndarray:
    """2D max pooling.

    x: (N, C, H, W) -> (N, C, H_out, W_out)
    stride varsayılan = kernel_size (overlapping yok).
    """
    x = _check_float32("x", x)
    N, C, H, W = x.shape
    k = kernel_size
    s = stride if stride is not None else k

    H_out = (H - k) // s + 1
    W_out = (W - k) // s + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"geçersiz pool çıktı: H_out={H_out} W_out={W_out}")

    out = np.empty((N, C, H_out, W_out), dtype=np.float32)
    for kh in range(k):
        for kw in range(k):
            patch = x[:, :, kh : kh + s * H_out : s, kw : kw + s * W_out : s]
            if kh == 0 and kw == 0:
                out = patch.copy()
            else:
                out = np.maximum(out, patch)
    return out
