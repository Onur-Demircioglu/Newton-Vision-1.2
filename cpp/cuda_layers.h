// newton-vision CUDA Hızlandırma Çekirdeği Header.
#pragma once
#include "layers.h"

namespace nv {

bool has_cuda_device();

std::vector<float> conv2d_cuda(const float* x, Shape4 xs,
                               const float* w, int C_out, int kH, int kW,
                               const float* bias,
                               int stride, int padding,
                               int& H_out, int& W_out);

std::vector<float> relu_cuda(const float* x, int64_t n);

std::vector<float> maxpool2d_cuda(const float* x, Shape4 xs,
                                  int kernel, int stride,
                                  int& H_out, int& W_out);

} // namespace nv
