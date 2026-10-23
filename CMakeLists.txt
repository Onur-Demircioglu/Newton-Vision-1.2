cmake_minimum_required(VERSION 3.17)
project(newton_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

include(CheckLanguage)
check_language(CUDA)

set(NV_CORE_SOURCES cpp/layers.cpp)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_definitions(-DHAS_CUDA)
    list(APPEND NV_CORE_SOURCES cpp/cuda_layers.cu)
endif()

# 1. Python Extension Target (_core)
pybind11_add_module(_core
    cpp/binding.cpp
    ${NV_CORE_SOURCES}
)

if(CMAKE_CUDA_COMPILER)
    set_property(TARGET _core PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(_core PRIVATE include cpp)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)

# 2. Pure C++ Standalone Native Library (libnewton_vision)
add_library(newton_vision_native STATIC ${NV_CORE_SOURCES})
target_include_directories(newton_vision_native PUBLIC include cpp)
target_link_libraries(newton_vision_native PUBLIC OpenMP::OpenMP_CXX)

# 3. Pure C++ Inference Demo Executable
add_executable(cpp_inference_demo examples/cpp_inference_demo.cpp)
target_link_libraries(cpp_inference_demo PRIVATE newton_vision_native OpenMP::OpenMP_CXX)

if(WIN32 AND MINGW)
    target_link_options(_core PRIVATE -static-libgcc -static-libstdc++)
    target_link_options(cpp_inference_demo PRIVATE -static-libgcc -static-libstdc++)
endif()

install(TARGETS _core DESTINATION newton_vision)
install(TARGETS newton_vision_native DESTINATION lib)
install(DIRECTORY include/newton DESTINATION include)
