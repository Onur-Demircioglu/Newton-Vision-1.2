cmake_minimum_required(VERSION 3.17)
project(newton_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# OpenMP: kod OpenMP 2.0 ile uyumludur (collapse yok, 'omp simd' MSVC'de makroyla kapalı). MSVC'de
# klasik /openmp (vcomp140, dağıtılabilir VC++ çalışma zamanı) kullanılır. /openmp:llvm (libomp140) aynı
# süreçte PyTorch'un Intel OpenMP'siyle (libiomp5md) çakışıp süreci sonlandırdığı için kullanılmaz.
find_package(OpenMP REQUIRED)
set(NV_OPENMP OpenMP::OpenMP_CXX)

include(CheckLanguage)
check_language(CUDA)

set(NV_CORE_SOURCES cpp/layers.cpp)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# 1. Python Extension Target (_core)
pybind11_add_module(_core
    cpp/binding.cpp
    ${NV_CORE_SOURCES}
)

if(CMAKE_CUDA_COMPILER)
    # CUDA yalnızca Python modülünde: conv2d / relu / maxpool2d forward çekirdekleri (cpp/cuda_layers.cu)
    target_sources(_core PRIVATE cpp/cuda_layers.cu)
    target_compile_definitions(_core PRIVATE HAS_CUDA)
    target_link_libraries(_core PRIVATE CUDA::cudart)
    set_property(TARGET _core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(_core PRIVATE include cpp)
if(WIN32 AND MINGW)
    # libgomp ve libwinpthread statik bağlanır. Aksi halde wheel, MinGW DLL'leri (libgomp-1.dll,
    # libwinpthread-1.dll) PATH'te olmayan makinelerde "DLL load failed" ile açılmaz.
    find_library(NV_GOMP_STATIC NAMES libgomp.a REQUIRED)
    find_library(NV_WINPTHREAD_STATIC NAMES libwinpthread.a REQUIRED)
    target_compile_options(_core PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(_core PRIVATE ${NV_GOMP_STATIC} ${NV_WINPTHREAD_STATIC})
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
else()
    target_link_libraries(_core PRIVATE ${NV_OPENMP})
endif()

# 2. Pure C++ Standalone Native Library (libnewton_vision)
add_library(newton_vision_native STATIC ${NV_CORE_SOURCES})
target_include_directories(newton_vision_native PUBLIC include cpp)
target_link_libraries(newton_vision_native PUBLIC ${NV_OPENMP})

# 3. Pure C++ Inference Demo Executable
add_executable(cpp_inference_demo examples/cpp_inference_demo.cpp)
target_link_libraries(cpp_inference_demo PRIVATE newton_vision_native ${NV_OPENMP})

# Not: Bağımsız C++ API'si (include/newton, newton_vision_native, demo) yalnızca CPU'dur. Başlıklardaki
# HAS_CUDA dalları var olmayan nv::cuda::* fonksiyonlarını çağırır; bu yüzden onlara HAS_CUDA tanımlanmaz.

if(MSVC)
    # Kaynak dosyalar UTF-8 (Türkçe karakterli yorum ve mesajlar)
    foreach(tgt _core newton_vision_native cpp_inference_demo)
        target_compile_options(${tgt} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
    endforeach()
endif()

if(WIN32 AND MINGW)
    target_link_options(cpp_inference_demo PRIVATE -static-libgcc -static-libstdc++)
endif()

install(TARGETS _core DESTINATION newton_vision)
install(TARGETS newton_vision_native DESTINATION lib)
install(DIRECTORY include/newton DESTINATION include)
