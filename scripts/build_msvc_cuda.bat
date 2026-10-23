@echo off
rem ============================================================================
rem newton-vision: MSVC + CUDA build'i (ayrı klasör: build\msvc-cuda)
rem
rem Varsayılan pip / MinGW build'ine dokunmaz. Çıktı: build\msvc-cuda\_core.cp313-win_amd64.pyd
rem CUDA 12.4, MSVC 14.51'i (Visual Studio 18 varsayılanı) desteklemez (cudafe++ çöker); bu yüzden
rem Visual Studio Installer'dan eklenen MSVC 14.44 araç seti kullanılır.
rem
rem Kullanım:   scripts\build_msvc_cuda.bat
rem Gerekirse aşağıdaki yolları kendi kurulumunuza göre değiştirin.
rem ============================================================================
setlocal
set "VS_ROOT=C:\Program Files\Microsoft Visual Studio\18\Community"
set "MSVC_VER=14.44"
set "WINSDK_VER=10.0.22621.0"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_ARCH=75"
set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
set "NINJA_EXE=C:\msys64\mingw64\bin\ninja.exe"

for /f "delims=" %%i in ('python -m pybind11 --cmakedir') do set "PYBIND11_DIR=%%i"

rem vcvars64 vswhere.exe'yi PATH'te arar
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\Installer;%PATH%"
rem vcvars, SDK algılama uyarısı yüzünden ortamı kurduğu halde hata kodu döndürebilir: cl.exe'yi doğrula
call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" %WINSDK_VER% -vcvars_ver=%MSVC_VER% >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 (echo [HATA] cl.exe bulunamadı: MSVC %MSVC_VER% araç seti kurulu mu? & exit /b 1)

rem vcvars bu makinede Windows SDK 'um' ve 'shared' yollarını eklemiyor: elle ekle
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "LIB=%LIB%;%SDK%\Lib\%WINSDK_VER%\um\x64"
set "INCLUDE=%INCLUDE%;%SDK%\Include\%WINSDK_VER%\um;%SDK%\Include\%WINSDK_VER%\shared"
set "PATH=%SDK%\bin\%WINSDK_VER%\x64;%CUDA_ROOT%\bin;%PATH%"
set "CUDA_PATH=%CUDA_ROOT%"

cd /d "%~dp0.."
"%CMAKE_EXE%" -S . -B build\msvc-cuda -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER="%CUDA_ROOT%\bin\nvcc.exe" ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCH% ^
  -Dpybind11_DIR="%PYBIND11_DIR%"
if errorlevel 1 (echo [HATA] CMake yapılandırması başarısız & exit /b 1)

"%CMAKE_EXE%" --build build\msvc-cuda
if errorlevel 1 (echo [HATA] Derleme başarısız & exit /b 1)

echo.
echo [OK] build\msvc-cuda\_core.cp313-win_amd64.pyd hazır.
echo      Kullanmak için CUDA_PATH tanımlı olmalı (cudart64_12.dll için), ör.:
echo      set CUDA_PATH=%CUDA_ROOT%
endlocal
