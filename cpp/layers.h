// newton-vision C++ çekirdeği — CNN & Transformer katmanları & Performance Engine.
// im2col + GEMM, OpenMP, Dilation, Grouped Conv, GELU, LayerNorm ve Autograd.
#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <cmath>

namespace nv {

// 4D tensör boyutu: (N, C, H, W)
struct Shape4 {
    int n, c, h, w;
    int64_t size() const { return int64_t(n) * c * h * w; }
};

inline void check_equal(int a, int b, const char* what) {
    if (a != b) throw std::invalid_argument(std::string(what) + ": " +
        std::to_string(a) + " != " + std::to_string(b));
}

// im2col dönüşümü (stride & dilation destekli)
void im2col_cpu(const float* data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_col);

// Paralel GEMM: C = A * B + (opsiyonel bias)
void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C, const float* bias = nullptr);

// Forward Pass Fonksiyonları
std::vector<float> conv2d(const float* x, Shape4 xs,
                          const float* w, int C_out, int kH, int kW,
                          const float* bias,
                          int stride, int padding, int dilation, int groups,
                          int& H_out, int& W_out);

std::vector<float> relu(const float* x, int64_t n);
std::vector<float> leaky_relu(const float* x, int64_t n, float negative_slope = 0.01f);
std::vector<float> sigmoid_fn(const float* x, int64_t n);
std::vector<float> tanh_fn(const float* x, int64_t n);
std::vector<float> gelu_fn(const float* x, int64_t n);

std::vector<float> maxpool2d(const float* x, Shape4 xs,
                             int kernel, int stride,
                             int& H_out, int& W_out);

std::vector<float> avgpool2d(const float* x, Shape4 xs,
                             int kernel, int stride,
                             int& H_out, int& W_out);

std::vector<float> batchnorm2d(const float* x, Shape4 xs,
                               const float* gamma, const float* beta,
                               const float* running_mean, const float* running_var,
                               float eps = 1e-5f);

std::vector<float> layernorm(const float* x, int num_samples, int feature_dim,
                             const float* gamma, const float* beta, float eps = 1e-5f);

// Backward Pass Fonksiyonları (Autograd Gradyan Motoru)
void conv2d_backward(const float* grad_y, const float* x, Shape4 xs,
                     const float* w, int C_out, int kH, int kW,
                     int stride, int padding, int dilation, int groups,
                     float* grad_x, float* grad_w, float* grad_bias);

void relu_backward(const float* grad_y, const float* x, int64_t n, float* grad_x);

void maxpool2d_backward(const float* grad_y, const float* x, Shape4 xs,
                        int kernel, int stride, float* grad_x);

} // namespace nv
