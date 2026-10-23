// newton-vision C++ çekirdeği — CNN katmanları & Performance Engine.
// im2col + GEMM, OpenMP, Dilation, Grouped Conv (Depthwise Separable) ve Autograd.
#include "layers.h"
#include "simd_gemm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nv {

namespace {
inline int64_t idx4(int n, int c, int h, int w, int C, int H, int Wd) {
    return ((int64_t)n * C + c) * H * Wd + h * Wd + w;
}
} // namespace

void im2col_cpu(const float* data_im, int channels, int height, int width,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_col) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int channels_col = channels * kernel_h * kernel_w;

    // Gruplu evrişimde dış döngü zaten paraleldir: iç içe iş parçacığı açma
    // (c, h) çiftleri tek düz döngüde: OpenMP 2.0 (MSVC /openmp) collapse desteklemez
    #pragma omp parallel for if(!omp_in_parallel())
    for (int ch = 0; ch < channels_col * height_col; ++ch) {
        { const int c = ch / height_col, h = ch % height_col;
            int w_offset = c % kernel_w;
            int h_offset = (c / kernel_w) % kernel_h;
            int c_im = c / (kernel_h * kernel_w);
            int im_row = h * stride_h + h_offset * dilation_h - pad_h;
            for (int w = 0; w < width_col; ++w) {
                int im_col = w * stride_w + w_offset * dilation_w - pad_w;
                int col_index = (c * height_col + h) * width_col + w;
                if (im_row >= 0 && im_row < height && im_col >= 0 && im_col < width) {
                    data_col[col_index] = data_im[(c_im * height + im_row) * width + im_col];
                } else {
                    data_col[col_index] = 0.0f;
                }
            }
        }
    }
}

static void col2im_cpu(const float* data_col, int channels, int height, int width,
                       int kernel_h, int kernel_w, int pad_h, int pad_w,
                       int stride_h, int stride_w, int dilation_h, int dilation_w,
                       float* data_im) {
    int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    // Her girdi kanalı yalnızca kendi düzlemine yazar: kanallar arası paralellik yarışsızdır.
    // Kanal içindeki toplama sırası (kh, kw, h, w) eski seri sürümle aynıdır.
    #pragma omp parallel for if(!omp_in_parallel())
    for (int c_im = 0; c_im < channels; ++c_im) {
        float* plane = data_im + static_cast<int64_t>(c_im) * height * width;
        std::fill(plane, plane + height * width, 0.0f);
        for (int kk = 0; kk < kernel_h * kernel_w; ++kk) {
            int c = c_im * kernel_h * kernel_w + kk;
            int w_offset = kk % kernel_w;
            int h_offset = kk / kernel_w;
            for (int h = 0; h < height_col; ++h) {
                int im_row = h * stride_h + h_offset * dilation_h - pad_h;
                if (im_row < 0 || im_row >= height) continue;
                for (int w = 0; w < width_col; ++w) {
                    int im_col = w * stride_w + w_offset * dilation_w - pad_w;
                    if (im_col >= 0 && im_col < width) {
                        plane[im_row * width + im_col] += data_col[(c * height_col + h) * width_col + w];
                    }
                }
            }
        }
    }
}

void gemm_cpu(int M, int N, int K, const float* A, const float* B, float* C, const float* bias) {
    gemm_avx2(M, N, K, A, B, C, bias);
}


std::vector<float> conv2d(const float* x, Shape4 xs,
                          const float* w, int C_out, int kH, int kW,
                          const float* bias,
                          int stride, int padding, int dilation, int groups,
                          int& H_out, int& W_out) {
    const int N = xs.n, C_in = xs.c, H = xs.h, W = xs.w;

    if (C_in % groups != 0 || C_out % groups != 0) {
        throw std::invalid_argument("conv2d: C_in ve C_out 'groups' değerine tam bölünmeli");
    }

    int eff_kH = dilation * (kH - 1) + 1;
    int eff_kW = dilation * (kW - 1) + 1;

    H_out = (H + 2 * padding - eff_kH) / stride + 1;
    W_out = (W + 2 * padding - eff_kW) / stride + 1;
    if (H_out <= 0 || W_out <= 0)
        throw std::invalid_argument("geçersiz conv çıktı boyutu");

    const int C_in_g = C_in / groups;
    const int C_out_g = C_out / groups;
    const int K = C_in_g * kH * kW;
    const int L = H_out * W_out;

    std::vector<float> y(int64_t(N) * C_out * L, 0.0f);

    auto run = [&](int n, int g, float* data_col) {
        const float* x_ng = x + (static_cast<int64_t>(n) * C_in + g * C_in_g) * H * W;
        const float* w_g = w + static_cast<int64_t>(g) * C_out_g * K;
        const float* b_g = bias ? (bias + g * C_out_g) : nullptr;
        float* y_ng = y.data() + (static_cast<int64_t>(n) * C_out + g * C_out_g) * L;

        im2col_cpu(x_ng, C_in_g, H, W, kH, kW, padding, padding, stride, stride, dilation, dilation, data_col);
        gemm_cpu(C_out_g, L, K, w_g, data_col, y_ng, b_g);
    };

    if (groups < omp_get_max_threads()) {
        // Az grup: paralellik im2col ve GEMM'in içinde
        std::vector<float> data_col(static_cast<size_t>(K) * L);
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < groups; ++g) run(n, g, data_col.data());
    } else {
        // Gruplu / depthwise: her (n, g) çifti bağımsız; grup başına küçük paralel bölgeler
        // açmak yerine (n, g) üzerinde paralelleştir, iş parçacığı başına ayrı tampon kullan
        #pragma omp parallel
        {
            std::vector<float> data_col(static_cast<size_t>(K) * L);
            #pragma omp for schedule(static)
            for (int ng = 0; ng < N * groups; ++ng)
                run(ng / groups, ng % groups, data_col.data());
        }
    }
    return y;
}

std::vector<float> relu(const float* x, int64_t n) {
    std::vector<float> y(n);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    return y;
}

std::vector<float> leaky_relu(const float* x, int64_t n, float negative_slope) {
    std::vector<float> y(n);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        y[i] = x[i] > 0.0f ? x[i] : x[i] * negative_slope;
    return y;
}

std::vector<float> sigmoid_fn(const float* x, int64_t n) {
    std::vector<float> y(n);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    return y;
}

std::vector<float> tanh_fn(const float* x, int64_t n) {
    std::vector<float> y(n);
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i)
        y[i] = std::tanh(x[i]);
    return y;
}

std::vector<float> maxpool2d(const float* x, Shape4 xs,
                             int kernel, int stride,
                             int& H_out, int& W_out) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    H_out = (H - kernel) / stride + 1;
    W_out = (W - kernel) / stride + 1;
    if (H_out <= 0 || W_out <= 0)
        throw std::invalid_argument("geçersiz maxpool çıktı boyutu");

    std::vector<float> y(int64_t(N) * C * H_out * W_out);

    #pragma omp parallel for
    for (int nc = 0; nc < N * C; ++nc) {
        { const int n = nc / C, c = nc % C;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            float v = x[idx4(n, c, ih, iw, C, H, W)];
                            if (v > max_val) max_val = v;
                        }
                    }
                    int64_t yidx = ((int64_t(n) * C + c) * H_out + oh) * W_out + ow;
                    y[yidx] = max_val;
                }
            }
        }
    }
    return y;
}

std::vector<float> avgpool2d(const float* x, Shape4 xs,
                             int kernel, int stride,
                             int& H_out, int& W_out) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    H_out = (H - kernel) / stride + 1;
    W_out = (W - kernel) / stride + 1;
    if (H_out <= 0 || W_out <= 0)
        throw std::invalid_argument("geçersiz avgpool çıktı boyutu");

    std::vector<float> y(int64_t(N) * C * H_out * W_out);
    const float inv_area = 1.0f / (kernel * kernel);

    #pragma omp parallel for
    for (int nc = 0; nc < N * C; ++nc) {
        { const int n = nc / C, c = nc % C;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float sum_val = 0.0f;
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            sum_val += x[idx4(n, c, ih, iw, C, H, W)];
                        }
                    }
                    int64_t yidx = ((int64_t(n) * C + c) * H_out + oh) * W_out + ow;
                    y[yidx] = sum_val * inv_area;
                }
            }
        }
    }
    return y;
}

std::vector<float> batchnorm2d(const float* x, Shape4 xs,
                               const float* gamma, const float* beta,
                               const float* running_mean, const float* running_var,
                               float eps) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    const int HW = H * W;
    std::vector<float> y(xs.size());

    #pragma omp parallel for
    for (int nc = 0; nc < N * C; ++nc) {
        { const int n = nc / C, c = nc % C;
            float mean = running_mean[c];
            float var = running_var[c];
            float g = gamma ? gamma[c] : 1.0f;
            float b = beta ? beta[c] : 0.0f;
            float inv_std = g / std::sqrt(var + eps);

            int64_t base_idx = ((int64_t(n) * C + c) * H) * W;
            for (int hw = 0; hw < HW; ++hw) {
                y[base_idx + hw] = (x[base_idx + hw] - mean) * inv_std + b;
            }
        }
    }
    return y;
}

void conv2d_backward(const float* grad_y, const float* x, Shape4 xs,
                     const float* w, int C_out, int kH, int kW,
                     int stride, int padding, int dilation, int groups,
                     float* grad_x, float* grad_w, float* grad_bias) {
    const int N = xs.n, C_in = xs.c, H = xs.h, W = xs.w;
    int eff_kH = dilation * (kH - 1) + 1;
    int eff_kW = dilation * (kW - 1) + 1;
    int H_out = (H + 2 * padding - eff_kH) / stride + 1;
    int W_out = (W + 2 * padding - eff_kW) / stride + 1;

    const int C_in_g = C_in / groups;
    const int C_out_g = C_out / groups;
    const int K = C_in_g * kH * kW;
    const int L = H_out * W_out;

    std::fill(grad_x, grad_x + xs.size(), 0.0f);
    std::fill(grad_w, grad_w + C_out * K, 0.0f);
    if (grad_bias) std::fill(grad_bias, grad_bias + C_out, 0.0f);

    // Tek (n, g) çifti için backward. gw/gb bu grubun kendi dilimine biriktirilir; gx (n, g)'ye özeldir.
    auto run = [&](int n, int g, float* data_col, float* grad_col) {
        const float* x_ng = x + (static_cast<int64_t>(n) * C_in + g * C_in_g) * H * W;
        const float* gy_ng = grad_y + (static_cast<int64_t>(n) * C_out + g * C_out_g) * L;
        const float* w_g = w + static_cast<int64_t>(g) * C_out_g * K;

        float* gx_ng = grad_x + (static_cast<int64_t>(n) * C_in + g * C_in_g) * H * W;
        float* gw_g = grad_w + static_cast<int64_t>(g) * C_out_g * K;
        float* gb_g = grad_bias ? (grad_bias + g * C_out_g) : nullptr;

        im2col_cpu(x_ng, C_in_g, H, W, kH, kW, padding, padding, stride, stride, dilation, dilation, data_col);

        if (gb_g) {
            for (int co = 0; co < C_out_g; ++co) {
                float sum_g = 0.0f;
                NV_OMP_SIMD_SUM(sum_g)
                for (int l = 0; l < L; ++l) sum_g += gy_ng[co * L + l];
                gb_g[co] += sum_g;
            }
        }

        // dW = dY · colᵀ   (C_out_g x K): her (co, k) bağımsız bir nokta çarpımı.
        // Tek düz döngü: OpenMP 2.0 (MSVC /openmp) collapse desteklemez.
        const int CK = C_out_g * K;
        #pragma omp parallel for if(!omp_in_parallel())
        for (int ck = 0; ck < CK; ++ck) {
            const int co = ck / K;
            const int k = ck % K;
            const float* gy_row = gy_ng + static_cast<int64_t>(co) * L;
            const float* col_row = data_col + static_cast<int64_t>(k) * L;
            float acc = 0.0f;
            NV_OMP_SIMD_SUM(acc)
            for (int l = 0; l < L; ++l) acc += gy_row[l] * col_row[l];
            gw_g[ck] += acc;
        }

        // dcol = Wᵀ · dY   (K x L): satır bazında, l ekseninde ardışık (vektörleşebilir) erişim
        #pragma omp parallel for if(!omp_in_parallel())
        for (int k = 0; k < K; ++k) {
            float* gc_row = grad_col + static_cast<int64_t>(k) * L;
            std::fill(gc_row, gc_row + L, 0.0f);
            for (int co = 0; co < C_out_g; ++co) {
                const float wv = w_g[co * K + k];
                const float* gy_row = gy_ng + static_cast<int64_t>(co) * L;
                NV_OMP_SIMD
                for (int l = 0; l < L; ++l) gc_row[l] += wv * gy_row[l];
            }
        }

        col2im_cpu(grad_col, C_in_g, H, W, kH, kW, padding, padding, stride, stride, dilation, dilation, gx_ng);
    };

    if (groups < omp_get_max_threads()) {
        // Az grup: (n, g) üzerinden seri (dW birikimi sıralı), iç döngüler paralel
        std::vector<float> data_col(static_cast<size_t>(K) * L);
        std::vector<float> grad_col(static_cast<size_t>(K) * L);
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < groups; ++g) run(n, g, data_col.data(), grad_col.data());
    } else {
        // Gruplu / depthwise: gruplar üzerinde paralel. Her iş parçacığı yalnızca kendi grubunun
        // dW/db dilimine yazar (n döngüsü grubun içinde seri) => yarış yok.
        #pragma omp parallel
        {
            std::vector<float> data_col(static_cast<size_t>(K) * L);
            std::vector<float> grad_col(static_cast<size_t>(K) * L);
            #pragma omp for schedule(static)
            for (int g = 0; g < groups; ++g)
                for (int n = 0; n < N; ++n)
                    run(n, g, data_col.data(), grad_col.data());
        }
    }
}

void relu_backward(const float* grad_y, const float* x, int64_t n, float* grad_x) {
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        grad_x[i] = x[i] > 0.0f ? grad_y[i] : 0.0f;
    }
}

void maxpool2d_backward(const float* grad_y, const float* x, Shape4 xs,
                        int kernel, int stride, float* grad_x) {
    const int N = xs.n, C = xs.c, H = xs.h, W = xs.w;
    int H_out = (H - kernel) / stride + 1;
    int W_out = (W - kernel) / stride + 1;

    std::fill(grad_x, grad_x + xs.size(), 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_ih = -1, max_iw = -1;
                    for (int kh = 0; kh < kernel; ++kh) {
                        for (int kw = 0; kw < kernel; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            float v = x[idx4(n, c, ih, iw, C, H, W)];
                            if (v > max_val) {
                                max_val = v;
                                max_ih = ih;
                                max_iw = iw;
                            }
                        }
                    }
                    if (max_ih >= 0 && max_iw >= 0) {
                        int64_t yidx = ((int64_t(n) * C + c) * H_out + oh) * W_out + ow;
                        int64_t xidx = idx4(n, c, max_ih, max_iw, C, H, W);
                        grad_x[xidx] += grad_y[yidx];
                    }
                }
            }
        }
    }
}

std::vector<float> gelu_fn(const float* x, int64_t n) {
    std::vector<float> y(n);
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        float xi = x[i];
        float inner = sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi);
        y[i] = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    return y;
}

std::vector<float> layernorm(const float* x, int num_samples, int feature_dim,
                             const float* gamma, const float* beta, float eps) {
    std::vector<float> y(static_cast<size_t>(num_samples) * feature_dim);
    #pragma omp parallel for
    for (int i = 0; i < num_samples; ++i) {
        const float* x_ptr = x + static_cast<int64_t>(i) * feature_dim;
        float* y_ptr = y.data() + static_cast<int64_t>(i) * feature_dim;

        double sum = 0.0;
        for (int j = 0; j < feature_dim; ++j) {
            sum += x_ptr[j];
        }
        float mean = static_cast<float>(sum / feature_dim);

        double var_sum = 0.0;
        for (int j = 0; j < feature_dim; ++j) {
            float diff = x_ptr[j] - mean;
            var_sum += diff * diff;
        }
        float var = static_cast<float>(var_sum / feature_dim);
        float std_inv = 1.0f / std::sqrt(var + eps);

        for (int j = 0; j < feature_dim; ++j) {
            float norm_val = (x_ptr[j] - mean) * std_inv;
            float g = gamma ? gamma[j] : 1.0f;
            float b = beta ? beta[j] : 0.0f;
            y_ptr[j] = norm_val * g + b;
        }
    }
    return y;
}

} // namespace nv

