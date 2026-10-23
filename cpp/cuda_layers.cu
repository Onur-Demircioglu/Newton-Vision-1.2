// newton-vision CUDA Hızlandırma Çekirdeği Impl.
//
// Her çağrı: host -> device kopya, çekirdek, device -> host kopya (kalıcı GPU belleği yok).
// Tüm CUDA çağrıları kontrol edilir; hata olursa std::runtime_error fırlatılır (Python'da RuntimeError).
#include "cuda_layers.h"
#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace nv {

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA hatası (") + what + "): " + cudaGetErrorString(err));
    }
}

// Hata durumunda da serbest bırakılan cihaz tamponu
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        if (bytes > 0) cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int kBlock = 256;

int grid_for(int64_t total) {
    int64_t blocks = (total + kBlock - 1) / kBlock;
    if (blocks > 2147483647LL) throw std::runtime_error("CUDA: çıktı çekirdek grid sınırını aşıyor");
    return static_cast<int>(blocks);
}

void launch_check(const char* what) {
    cuda_check(cudaGetLastError(), what);
    cuda_check(cudaDeviceSynchronize(), what);
}

} // namespace

// Bir iş parçacığı = bir çıktı elemanı (n, co, oh, ow). Gruplu ve dilated evrişimi destekler;
// CPU im2col yoluyla aynı indeksleme: giriş kanalı = g * C_in_g + ci, ih = oh*s + kh*d - p.
__global__ void conv2d_kernel(const float* x, const float* w, const float* bias,
                              int N, int C_in, int H, int W,
                              int C_out, int kH, int kW,
                              int stride, int padding, int dilation, int groups,
                              int H_out, int W_out, float* y) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * C_out * H_out * W_out;
    if (idx >= total) return;

    int ow = static_cast<int>(idx % W_out);
    int oh = static_cast<int>((idx / W_out) % H_out);
    int co = static_cast<int>((idx / (static_cast<int64_t>(W_out) * H_out)) % C_out);
    int n = static_cast<int>(idx / (static_cast<int64_t>(W_out) * H_out * C_out));

    const int C_in_g = C_in / groups;
    const int C_out_g = C_out / groups;
    const int g = co / C_out_g;

    float acc = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in_g; ++ci) {
        const int c = g * C_in_g + ci;
        for (int kh = 0; kh < kH; ++kh) {
            const int ih = oh * stride + kh * dilation - padding;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                const int iw = ow * stride + kw * dilation - padding;
                if (iw < 0 || iw >= W) continue;
                const int64_t xidx = ((static_cast<int64_t>(n) * C_in + c) * H + ih) * W + iw;
                const int64_t widx = ((static_cast<int64_t>(co) * C_in_g + ci) * kH + kh) * kW + kw;
                acc += x[xidx] * w[widx];
            }
        }
    }
    y[idx] = acc;
}

__global__ void relu_kernel(const float* x, float* y, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

__global__ void maxpool2d_kernel(const float* x, int N, int C, int H, int W,
                                 int kernel, int stride, int H_out, int W_out, float* y) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    if (idx >= total) return;

    int ow = static_cast<int>(idx % W_out);
    int oh = static_cast<int>((idx / W_out) % H_out);
    int64_t nc = idx / (static_cast<int64_t>(W_out) * H_out);   // n * C + c

    // CPU çekirdeğiyle aynı: -inf başlangıç, katı '>' karşılaştırması
    float max_val = -INFINITY;
    for (int kh = 0; kh < kernel; ++kh) {
        for (int kw = 0; kw < kernel; ++kw) {
            const int ih = oh * stride + kh;
            const int iw = ow * stride + kw;
            const float v = x[(nc * H + ih) * W + iw];
            if (v > max_val) max_val = v;
        }
    }
    y[idx] = max_val;
}

bool has_cuda_device() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

std::vector<float> conv2d_cuda(const float* x, Shape4 xs,
                               const float* w, int C_out, int kH, int kW,
                               const float* bias,
                               int stride, int padding, int dilation, int groups,
                               int& H_out, int& W_out) {
    const int N = xs.n, C_in = xs.c, H = xs.h, W = xs.w;
    if (groups < 1 || C_in % groups != 0 || C_out % groups != 0) {
        throw std::invalid_argument("conv2d_cuda: C_in ve C_out 'groups' değerine tam bölünmeli");
    }
    const int eff_kH = dilation * (kH - 1) + 1;
    const int eff_kW = dilation * (kW - 1) + 1;
    H_out = (H + 2 * padding - eff_kH) / stride + 1;
    W_out = (W + 2 * padding - eff_kW) / stride + 1;
    if (H_out <= 0 || W_out <= 0) throw std::invalid_argument("geçersiz conv çıktı boyutu");

    const size_t x_bytes = static_cast<size_t>(xs.size()) * sizeof(float);
    const size_t w_bytes = static_cast<size_t>(C_out) * (C_in / groups) * kH * kW * sizeof(float);
    const size_t b_bytes = bias ? static_cast<size_t>(C_out) * sizeof(float) : 0;
    const int64_t total = static_cast<int64_t>(N) * C_out * H_out * W_out;
    const size_t y_bytes = static_cast<size_t>(total) * sizeof(float);

    DeviceBuffer d_x(x_bytes), d_w(w_bytes), d_b(b_bytes), d_y(y_bytes);
    cuda_check(cudaMemcpy(d_x.ptr, x, x_bytes, cudaMemcpyHostToDevice), "conv2d x kopyası");
    cuda_check(cudaMemcpy(d_w.ptr, w, w_bytes, cudaMemcpyHostToDevice), "conv2d w kopyası");
    if (bias) cuda_check(cudaMemcpy(d_b.ptr, bias, b_bytes, cudaMemcpyHostToDevice), "conv2d bias kopyası");

    conv2d_kernel<<<grid_for(total), kBlock>>>(d_x.ptr, d_w.ptr, bias ? d_b.ptr : nullptr,
                                               N, C_in, H, W, C_out, kH, kW,
                                               stride, padding, dilation, groups, H_out, W_out, d_y.ptr);
    launch_check("conv2d_kernel");

    std::vector<float> y(static_cast<size_t>(total));
    cuda_check(cudaMemcpy(y.data(), d_y.ptr, y_bytes, cudaMemcpyDeviceToHost), "conv2d y kopyası");
    return y;
}

std::vector<float> relu_cuda(const float* x, int64_t n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    DeviceBuffer d_x(bytes), d_y(bytes);
    cuda_check(cudaMemcpy(d_x.ptr, x, bytes, cudaMemcpyHostToDevice), "relu x kopyası");

    relu_kernel<<<grid_for(n), kBlock>>>(d_x.ptr, d_y.ptr, n);
    launch_check("relu_kernel");

    std::vector<float> y(static_cast<size_t>(n));
    cuda_check(cudaMemcpy(y.data(), d_y.ptr, bytes, cudaMemcpyDeviceToHost), "relu y kopyası");
    return y;
}

std::vector<float> maxpool2d_cuda(const float* x, Shape4 xs,
                                  int kernel, int stride,
                                  int& H_out, int& W_out) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    H_out = (H - kernel) / stride + 1;
    W_out = (W - kernel) / stride + 1;
    if (H_out <= 0 || W_out <= 0) throw std::invalid_argument("geçersiz maxpool çıktı boyutu");

    const size_t x_bytes = static_cast<size_t>(xs.size()) * sizeof(float);
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    const size_t y_bytes = static_cast<size_t>(total) * sizeof(float);

    DeviceBuffer d_x(x_bytes), d_y(y_bytes);
    cuda_check(cudaMemcpy(d_x.ptr, x, x_bytes, cudaMemcpyHostToDevice), "maxpool x kopyası");

    maxpool2d_kernel<<<grid_for(total), kBlock>>>(d_x.ptr, N, C, H, W, kernel, stride, H_out, W_out, d_y.ptr);
    launch_check("maxpool2d_kernel");

    std::vector<float> y(static_cast<size_t>(total));
    cuda_check(cudaMemcpy(y.data(), d_y.ptr, y_bytes, cudaMemcpyDeviceToHost), "maxpool y kopyası");
    return y;
}

} // namespace nv
