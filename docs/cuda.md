# CUDA Build'i

Varsayılan pip / MinGW build'i yalnızca CPU'dur. Bu belge Windows'ta MSVC + CUDA build'ini ve kapsamını anlatır.

## Derleme (Windows, MSVC + CUDA)

```bat
scripts\build_msvc_cuda.bat
```

- **Çıktı:** `build\msvc-cuda\_core.cp313-win_amd64.pyd`. Varsayılan MinGW build'ine dokunmaz.
- **Gereksinimler:** CUDA 12.4 ve Visual Studio'da **MSVC 14.44** araç seti. CUDA 12.4, VS 18'in varsayılan MSVC 14.51'iyle derlerken çöker; script bu yüzden 14.44'ü seçer. Yollar scriptin başında tanımlıdır.
- **Çalışma zamanı:** `cudart64_12.dll` için `CUDA_PATH` tanımlı olmalı. newton-vision onu `os.add_dll_directory` ile ekler.
- **Kapsam:**
  - CUDA'da çalışan işlemler: `conv2d` (stride, padding, dilation, groups), `relu` ve `maxpool2d`; hepsi yalnızca **forward**.
  - Her çağrıda host ↔ device kopyası yapılır. Diğer tüm işlemler ve tüm backward hesapları CPU'dadır.
  - Doğrulama: `tests/test_cuda_kernels.py` çekirdekleri ve ResNet-18 / MobileNetV2'nin tam forward'ını CPU ile karşılaştırır.
  - Bağımsız C++ API'si (`include/newton`) yalnızca CPU'dur.
- **Hız:** GTX 1650'de büyük bir evrişim (8×64×56×56, 3×3) CPU'da 67.6 ms, CUDA'da 19.1 ms sürdü. Bu build'in CPU yolu MinGW build'inden yavaştır, çünkü MSVC'de `omp simd` yok.

**AVX2 SIMD Tespiti:**
C++ backend çalışma zamanında `CPUID` komutu ile CPU'nun AVX2 desteğini kontrol eder. AVX2 varsa SIMD hızlandırılmış GEMM kullanılır; yoksa fallback yoluna geçilir. Bu sayede ARM (Apple Silicon) veya eski x86 işlemcilerde `SIGILL` çökmesi yaşanmaz.
