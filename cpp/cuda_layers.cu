// newton-vision CUDA Hızlandırma Çekirdeği Impl.
#include "cuda_layers.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

namespace nv {

__global__ void conv2d_kernel(const float* x, const float* w, const float* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride, int padding,
                              int H_out, int W_out,
                              float* y) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int index_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow >= W_out || oh >= H_out) return;

    int n = index_z / C_out;
    int co = index_z % C_out;

    if (n >= N || co >= C_out) return;

    float acc = bias ? bias[co] : 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int64_t xidx = ((int64_t(n) * C_in + ci) * H + ih) * W + iw;
                    int64_t widx = ((int64_t(co) * C_in + ci) * kH + kh) * kW + kw;
                    acc += x[xidx] * w[widx];
                }
            }
        }
    }

    int64_t yidx = ((int64_t(n) * C_out + co) * H_out + oh) * W_out + ow;
    y[yidx] = acc;
}

__global__ void relu_kernel(const float* x, float* y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void maxpool2d_kernel(const float* x, int N, int C, int H, int W,
                                 int kernel, int stride, int H_out, int W_out,
                                 float* y) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int index_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow >= W_out || oh >= H_out) return;

    int n = index_z / C;
    int c = index_z % C;

    if (n >= N || c >= C) return;

    float max_val = -1e38f;
    for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            if (ih < H && iw < W) {
                int64_t xidx = ((int64_t(n) * C + c) * H + ih) * W + iw;
                float v = x[xidx];
                if (v > max_val) max_val = v;
            }
        }
    }

    int64_t yidx = ((int64_t(n) * C + c) * H_out + oh) * W_out + ow;
    y[yidx] = max_val;
}

bool has_cuda_device() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

std::vector<float> conv2d_cuda(const float* x, Shape4 xs,
                               const float* w, int C_out, int kH, int kW,
                               const float* bias,
                               int stride, int padding,
                               int& H_out, int& W_out) {
    const int N = xs.n, C_in = xs.c, H = xs.h, W = xs.w;
    H_out = (H + 2 * padding - kH) / stride + 1;
    W_out = (W + 2 * padding - kW) / stride + 1;

    size_t x_size = xs.size() * sizeof(float);
    size_t w_size = static_cast<size_t>(C_out) * C_in * kH * kW * sizeof(float);
    size_t b_size = bias ? C_out * sizeof(float) : 0;
    size_t y_size = static_cast<size_t>(N) * C_out * H_out * W_out * sizeof(float);

    float *d_x = nullptr, *d_w = nullptr, *d_b = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, x_size);
    cudaMalloc(&d_w, w_size);
    if (bias) cudaMalloc(&d_b, b_size);
    cudaMalloc(&d_y, y_size);

    cudaMemcpy(d_x, x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, w_size, cudaMemcpyHostToDevice);
    if (bias) cudaMemcpy(d_b, bias, b_size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                 (H_out + blockDim.y - 1) / blockDim.y,
                 N * C_out);

    conv2d_kernel<<<gridDim, blockDim>>>(d_x, d_w, d_b, N, C_in, H, W, C_out, kH, kW, stride, padding, H_out, W_out, d_y);
    cudaDeviceSynchronize();

    std::vector<float> y(static_cast<size_t>(N) * C_out * H_out * W_out);
    cudaMemcpy(y.data(), d_y, y_size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_w);
    if (d_b) cudaFree(d_b);
    cudaFree(d_y);

    return y;
}

std::vector<float> relu_cuda(const float* x, int64_t n) {
    size_t bytes = n * sizeof(float);
    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);

    relu_kernel<<<grid, block>>>(d_x, d_y, n);
    cudaDeviceSynchronize();

    std::vector<float> y(n);
    cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

std::vector<float> maxpool2d_cuda(const float* x, Shape4 xs,
                                  int kernel, int stride,
                                  int& H_out, int& W_out) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    H_out = (H - kernel) / stride + 1;
    W_out = (W - kernel) / stride + 1;

    size_t x_bytes = xs.size() * sizeof(float);
    size_t y_bytes = static_cast<size_t>(N) * C * H_out * W_out * sizeof(float);

    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_y, y_bytes);

    cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim((W_out + blockDim.x - 1) / blockDim.x,
                 (H_out + blockDim.y - 1) / blockDim.y,
                 N * C);

    maxpool2d_kernel<<<gridDim, blockDim>>>(d_x, N, C, H, W, kernel, stride, H_out, W_out, d_y);
    cudaDeviceSynchronize();

    std::vector<float> y(static_cast<size_t>(N) * C * H_out * W_out);
    cudaMemcpy(y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    return y;
}

} // namespace nv
