#pragma once
#include "newton/tensor.hpp"
#include "../../cpp/layers.h"

#ifdef HAS_CUDA
#include "../../cpp/cuda_layers.h"
#endif

namespace nv {

class Layer {
public:
    virtual ~Layer() = default;
    virtual Tensor forward(const Tensor& input) = 0;

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }
};

class Conv2d : public Layer {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int groups;

    Tensor weight;
    Tensor bias;
    bool has_bias;

    Conv2d(int in_ch, int out_ch, int k_size, int st = 1, int pad = 0, int dil = 1, int grp = 1, bool b = true)
        : in_channels(in_ch), out_channels(out_ch), kernel_size(k_size),
          stride(st), padding(pad), dilation(dil), groups(grp), has_bias(b) {
        
        Shape w_shape{out_channels, in_channels / groups, kernel_size, kernel_size};
        weight = Tensor(w_shape);
        if (has_bias) {
            bias = Tensor(Shape{1, out_channels, 1, 1});
        }
    }

    Tensor forward(const Tensor& input) override {
        const Shape& in_shape = input.shape();
        int H_out = 0, W_out = 0;

        Shape4 xs{in_shape.n, in_shape.c, in_shape.h, in_shape.w};

        if (input.device() == Device::CPU) {
            const float* bias_ptr = has_bias ? bias.data() : nullptr;
            std::vector<float> out_vec = nv::conv2d(
                input.data(), xs,
                weight.data(), out_channels, kernel_size, kernel_size,
                bias_ptr, stride, padding, dilation, groups,
                H_out, W_out
            );
            Shape out_shape{in_shape.n, out_channels, H_out, W_out};
            return Tensor(out_shape, out_vec, Device::CPU);
        } else {
#ifdef HAS_CUDA
            H_out = (in_shape.h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            W_out = (in_shape.w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
            Shape out_shape{in_shape.n, out_channels, H_out, W_out};
            Tensor output(out_shape, Device::CUDA);

            const float* bias_ptr = has_bias ? bias.data() : nullptr;
            nv::cuda::conv2d_forward(
                input.data(), in_shape.n, in_shape.c, in_shape.h, in_shape.w,
                weight.data(), out_channels, kernel_size, kernel_size,
                bias_ptr, stride, padding, dilation, groups,
                output.data()
            );
            return output;
#else
            throw std::runtime_error("CUDA is not enabled in this C++ build.");
#endif
        }
    }
};

class Linear : public Layer {
public:
    int in_features;
    int out_features;
    Tensor weight; // Shape: (out_features, in_features)
    Tensor bias;   // Shape: (1, out_features, 1, 1)
    bool has_bias;

    Linear(int in_f, int out_f, bool b = true)
        : in_features(in_f), out_features(out_f), has_bias(b) {
        weight = Tensor(Shape{out_features, in_features, 1, 1});
        if (has_bias) {
            bias = Tensor(Shape{1, out_features, 1, 1});
        }
    }

    Tensor forward(const Tensor& input) override {
        const Shape& in_shape = input.shape();
        int batch_size = in_shape.n;

        Shape out_shape{batch_size, out_features, 1, 1};

        if (input.device() == Device::CPU) {
            Tensor output(out_shape, Device::CPU);
            const float* bias_ptr = has_bias ? bias.data() : nullptr;
            nv::gemm_cpu(batch_size, out_features, in_features,
                          input.data(), weight.data(), output.data(), bias_ptr);
            return output;
        } else {
#ifdef HAS_CUDA
            Tensor output(out_shape, Device::CUDA);
            Tensor input_cpu = input;
            input_cpu.to(Device::CPU);
            Tensor out_cpu = forward(input_cpu);
            out_cpu.to(Device::CUDA);
            return out_cpu;
#else
            throw std::runtime_error("CUDA not supported.");
#endif
        }
    }
};

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        const Shape& shape = input.shape();
        if (input.device() == Device::CPU) {
            std::vector<float> out_vec = nv::relu(input.data(), shape.size());
            return Tensor(shape, out_vec, Device::CPU);
        } else {
#ifdef HAS_CUDA
            Tensor output(shape, Device::CUDA);
            nv::cuda::relu_forward(input.data(), output.data(), shape.size());
            return output;
#else
            throw std::runtime_error("CUDA not supported.");
#endif
        }
    }
};

class MaxPool2d : public Layer {
public:
    int kernel_size;
    int stride;

    MaxPool2d(int k_size, int st = 0)
        : kernel_size(k_size), stride(st == 0 ? k_size : st) {}

    Tensor forward(const Tensor& input) override {
        const Shape& in_shape = input.shape();
        int H_out = 0, W_out = 0;
        Shape4 xs{in_shape.n, in_shape.c, in_shape.h, in_shape.w};

        if (input.device() == Device::CPU) {
            std::vector<float> out_vec = nv::maxpool2d(input.data(), xs, kernel_size, stride, H_out, W_out);
            Shape out_shape{in_shape.n, in_shape.c, H_out, W_out};
            return Tensor(out_shape, out_vec, Device::CPU);
        } else {
#ifdef HAS_CUDA
            H_out = (in_shape.h - kernel_size) / stride + 1;
            W_out = (in_shape.w - kernel_size) / stride + 1;
            Shape out_shape{in_shape.n, in_shape.c, H_out, W_out};
            Tensor output(out_shape, Device::CUDA);
            nv::cuda::maxpool2d_forward(input.data(), in_shape.n, in_shape.c, in_shape.h, in_shape.w,
                                         kernel_size, stride, output.data());
            return output;
#else
            throw std::runtime_error("CUDA not supported.");
#endif
        }
    }
};

class BatchNorm2d : public Layer {
public:
    int num_features;
    float eps;
    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;

    BatchNorm2d(int num_feat, float epsilon = 1e-5f)
        : num_features(num_feat), eps(epsilon) {
        gamma = Tensor(Shape{1, num_features, 1, 1});
        beta = Tensor(Shape{1, num_features, 1, 1});
        running_mean = Tensor(Shape{1, num_features, 1, 1});
        running_var = Tensor(Shape{1, num_features, 1, 1});
    }

    Tensor forward(const Tensor& input) override {
        const Shape& in_shape = input.shape();
        Shape4 xs{in_shape.n, in_shape.c, in_shape.h, in_shape.w};

        if (input.device() == Device::CPU) {
            std::vector<float> out_vec = nv::batchnorm2d(
                input.data(), xs, gamma.data(), beta.data(),
                running_mean.data(), running_var.data(), eps
            );
            return Tensor(in_shape, out_vec, Device::CPU);
        } else {
            Tensor input_cpu = input;
            input_cpu.to(Device::CPU);
            Tensor out_cpu = forward(input_cpu);
            out_cpu.to(Device::CUDA);
            return out_cpu;
        }
    }
};

} // namespace nv
