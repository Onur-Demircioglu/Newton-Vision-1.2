"""Ref self-test: ebatlar + bilinen değerlerle doğrulama."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from layers import conv2d, relu, maxpool2d

np.random.seed(0)

# --- conv2d: küçük elle hesaplanabilir örnek ---
# N=1, C_in=1, H=W=3; çekirdek 2x2 full-of-1 ile toplam; stride=1, pad=0
x = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)  # 1..9
w = np.ones((1, 1, 2, 2), dtype=np.float32)
y = conv2d(x, w, stride=1, padding=0)
print("conv(x,w) boyut:", y.shape, " beklenen (1,1,2,2)")
# Pencere toplamları: [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12,16,24,28]
print("conv sonuc:", y.ravel())
assert np.allclose(y.ravel(), [12, 16, 24, 28]), "conv yanlış!"

# --- padding + stride + bias + çok kanal ---
x2 = np.random.randn(2, 3, 8, 8).astype(np.float32)
w2 = np.random.randn(5, 3, 3, 3).astype(np.float32)
b2 = np.random.randn(5).astype(np.float32)
y2 = conv2d(x2, w2, bias=b2, stride=2, padding=1)
# (8+2-3)//2+1 = 4
print("conv çok kanal boyut:", y2.shape, " beklenen (2,5,4,4)")
assert y2.shape == (2, 5, 4, 4), "boyut yanlış!"

# --- relu ---
xr = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
print("relu:", relu(xr))
assert np.array_equal(relu(xr), [0, 0, 0, 1, 2]), "relu yanlış"

# --- maxpool2d ---
xp = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
mp = maxpool2d(xp, kernel_size=2)
print("maxpool boyut:", mp.shape, " beklenen (1,1,2,2)")
# 2x2 non-overlapping (stride=2) blok max'lari:
#   sol-ust [0,1,4,5]->5, sag-ust [2,3,6,7]->7,
#   sol-alt [8,9,12,13]->13, sag-alt [10,11,14,15]->15
print("maxpool sonuc:", mp.ravel())
assert np.array_equal(mp.ravel(), [5, 7, 13, 15]), "maxpool yanlis"

print("\n✅ Tüm ref self-testler geçti.")
