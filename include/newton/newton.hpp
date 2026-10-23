#pragma once
/**
 * 🌌 newton-vision: C++17 & CUDA Native Framework Header
 * ======================================================
 * High-Performance Computer Vision & Deep Learning Engine in Pure C++.
 */

#include "newton/tensor.hpp"
#include "newton/layers.hpp"
#include "newton/model.hpp"

namespace nv {

inline bool is_cuda_available() {
#ifdef HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

} // namespace nv
