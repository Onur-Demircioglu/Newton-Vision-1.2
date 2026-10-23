// newton-vision C++ pybind11 köprüsü — Forward, Backward, CUDA Engine & Groups/Dilation
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "layers.h"
#include "simd_gemm.h"

#ifdef HAS_CUDA
#include "cuda_layers.h"
#endif

#include <optional>
#include <stdexcept>
#include <vector>
#include <tuple>

namespace py = pybind11;

bool has_cuda_py() {
#ifdef HAS_CUDA
    return nv::has_cuda_device();
#else
    return false;
#endif
}

py::array_t<float> conv2d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> w,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> bias,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    bool use_cuda = false
) {
    py::buffer_info x_info = x.request();
    py::buffer_info w_info = w.request();

    if (x_info.ndim != 4 || w_info.ndim != 4) {
        throw std::invalid_argument("conv2d: x ve w 4D tensör olmalı");
    }

    int N = static_cast<int>(x_info.shape[0]);
    int C_in = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);

    int C_out = static_cast<int>(w_info.shape[0]);
    int kH = static_cast<int>(w_info.shape[2]);
    int kW = static_cast<int>(w_info.shape[3]);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        py::buffer_info b_info = bias->request();
        bias_ptr = static_cast<const float*>(b_info.ptr);
    }

    nv::Shape4 xs{N, C_in, H, W};
    int H_out = 0, W_out = 0;

    const float* x_ptr = static_cast<const float*>(x_info.ptr);
    const float* w_ptr = static_cast<const float*>(w_info.ptr);

    std::vector<float> y_vec;

#ifdef HAS_CUDA
    if (use_cuda && nv::has_cuda_device()) {
        y_vec = nv::conv2d_cuda(x_ptr, xs, w_ptr, C_out, kH, kW, bias_ptr, stride, padding, dilation, groups, H_out, W_out);
    } else {
        y_vec = nv::conv2d(x_ptr, xs, w_ptr, C_out, kH, kW, bias_ptr, stride, padding, dilation, groups, H_out, W_out);
    }
#else
    y_vec = nv::conv2d(x_ptr, xs, w_ptr, C_out, kH, kW, bias_ptr, stride, padding, dilation, groups, H_out, W_out);
#endif

    py::array_t<float> result({N, C_out, H_out, W_out});
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));

    return result;
}

std::tuple<py::array_t<float>, py::array_t<float>, std::optional<py::array_t<float>>> conv2d_backward_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> grad_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> w,
    bool calc_bias,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1
) {
    py::buffer_info gy_info = grad_y.request();
    py::buffer_info x_info = x.request();
    py::buffer_info w_info = w.request();

    int N = static_cast<int>(x_info.shape[0]);
    int C_in = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);

    int C_out = static_cast<int>(w_info.shape[0]);
    int kH = static_cast<int>(w_info.shape[2]);
    int kW = static_cast<int>(w_info.shape[3]);

    nv::Shape4 xs{N, C_in, H, W};

    py::array_t<float> grad_x(x_info.shape);
    py::array_t<float> grad_w(w_info.shape);
    std::optional<py::array_t<float>> grad_bias = std::nullopt;

    float* gx_ptr = static_cast<float*>(grad_x.request().ptr);
    float* gw_ptr = static_cast<float*>(grad_w.request().ptr);
    float* gb_ptr = nullptr;

    if (calc_bias) {
        grad_bias = py::array_t<float>({C_out});
        gb_ptr = static_cast<float*>(grad_bias->request().ptr);
    }

    nv::conv2d_backward(
        static_cast<const float*>(gy_info.ptr),
        static_cast<const float*>(x_info.ptr),
        xs,
        static_cast<const float*>(w_info.ptr),
        C_out, kH, kW, stride, padding, dilation, groups,
        gx_ptr, gw_ptr, gb_ptr
    );

    return std::make_tuple(grad_x, grad_w, grad_bias);
}

py::array_t<float> relu_py(py::array_t<float, py::array::c_style | py::array::forcecast> x, bool use_cuda = false) {
    py::buffer_info x_info = x.request();
    std::vector<float> y_vec;
#ifdef HAS_CUDA
    if (use_cuda && nv::has_cuda_device()) {
        y_vec = nv::relu_cuda(static_cast<const float*>(x_info.ptr), x_info.size);
    } else {
        y_vec = nv::relu(static_cast<const float*>(x_info.ptr), x_info.size);
    }
#else
    y_vec = nv::relu(static_cast<const float*>(x_info.ptr), x_info.size);
#endif
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> relu_backward_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> grad_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    py::buffer_info gy_info = grad_y.request();
    py::buffer_info x_info = x.request();
    py::array_t<float> grad_x(x_info.shape);
    nv::relu_backward(
        static_cast<const float*>(gy_info.ptr),
        static_cast<const float*>(x_info.ptr),
        x_info.size,
        static_cast<float*>(grad_x.request().ptr)
    );
    return grad_x;
}

py::array_t<float> leaky_relu_py(py::array_t<float, py::array::c_style | py::array::forcecast> x, float slope = 0.01f) {
    py::buffer_info x_info = x.request();
    std::vector<float> y_vec = nv::leaky_relu(static_cast<const float*>(x_info.ptr), x_info.size, slope);
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> sigmoid_py(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    py::buffer_info x_info = x.request();
    std::vector<float> y_vec = nv::sigmoid_fn(static_cast<const float*>(x_info.ptr), x_info.size);
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> tanh_py(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    py::buffer_info x_info = x.request();
    std::vector<float> y_vec = nv::tanh_fn(static_cast<const float*>(x_info.ptr), x_info.size);
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> maxpool2d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int kernel_size,
    std::optional<int> stride,
    bool use_cuda = false
) {
    py::buffer_info x_info = x.request();
    int N = static_cast<int>(x_info.shape[0]);
    int C = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);
    int s = stride.value_or(kernel_size);
    int H_out = 0, W_out = 0;
    std::vector<float> y_vec;
#ifdef HAS_CUDA
    if (use_cuda && nv::has_cuda_device()) {
        y_vec = nv::maxpool2d_cuda(static_cast<const float*>(x_info.ptr), {N, C, H, W}, kernel_size, s, H_out, W_out);
    } else {
        y_vec = nv::maxpool2d(static_cast<const float*>(x_info.ptr), {N, C, H, W}, kernel_size, s, H_out, W_out);
    }
#else
    y_vec = nv::maxpool2d(static_cast<const float*>(x_info.ptr), {N, C, H, W}, kernel_size, s, H_out, W_out);
#endif
    py::array_t<float> result({N, C, H_out, W_out});
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> maxpool2d_backward_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> grad_y,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int kernel_size,
    std::optional<int> stride
) {
    py::buffer_info gy_info = grad_y.request();
    py::buffer_info x_info = x.request();
    int N = static_cast<int>(x_info.shape[0]);
    int C = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);
    int s = stride.value_or(kernel_size);

    py::array_t<float> grad_x(x_info.shape);
    nv::maxpool2d_backward(
        static_cast<const float*>(gy_info.ptr),
        static_cast<const float*>(x_info.ptr),
        {N, C, H, W}, kernel_size, s,
        static_cast<float*>(grad_x.request().ptr)
    );
    return grad_x;
}

py::array_t<float> avgpool2d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    int kernel_size,
    std::optional<int> stride
) {
    py::buffer_info x_info = x.request();
    int N = static_cast<int>(x_info.shape[0]);
    int C = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);
    int s = stride.value_or(kernel_size);
    int H_out = 0, W_out = 0;
    std::vector<float> y_vec = nv::avgpool2d(static_cast<const float*>(x_info.ptr), {N, C, H, W}, kernel_size, s, H_out, W_out);
    py::array_t<float> result({N, C, H_out, W_out});
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> batchnorm2d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> gamma,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> beta,
    py::array_t<float, py::array::c_style | py::array::forcecast> running_mean,
    py::array_t<float, py::array::c_style | py::array::forcecast> running_var,
    float eps = 1e-5f
) {
    py::buffer_info x_info = x.request();
    int N = static_cast<int>(x_info.shape[0]);
    int C = static_cast<int>(x_info.shape[1]);
    int H = static_cast<int>(x_info.shape[2]);
    int W = static_cast<int>(x_info.shape[3]);
    const float* g_ptr = gamma.has_value() ? static_cast<const float*>(gamma->request().ptr) : nullptr;
    const float* b_ptr = beta.has_value() ? static_cast<const float*>(beta->request().ptr) : nullptr;

    std::vector<float> y_vec = nv::batchnorm2d(
        static_cast<const float*>(x_info.ptr), {N, C, H, W}, g_ptr, b_ptr,
        static_cast<const float*>(running_mean.request().ptr),
        static_cast<const float*>(running_var.request().ptr), eps
    );
    py::array_t<float> result({N, C, H, W});
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> gelu_py(py::array_t<float, py::array::c_style | py::array::forcecast> x) {
    py::buffer_info x_info = x.request();
    std::vector<float> y_vec = nv::gelu_fn(static_cast<const float*>(x_info.ptr), x_info.size);
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

py::array_t<float> layernorm_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> gamma,
    std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> beta,
    float eps = 1e-5f
) {
    py::buffer_info x_info = x.request();
    int feature_dim = static_cast<int>(x_info.shape[x_info.ndim - 1]);
    int num_samples = static_cast<int>(x_info.size / feature_dim);

    const float* g_ptr = gamma.has_value() ? static_cast<const float*>(gamma->request().ptr) : nullptr;
    const float* b_ptr = beta.has_value() ? static_cast<const float*>(beta->request().ptr) : nullptr;

    std::vector<float> y_vec = nv::layernorm(static_cast<const float*>(x_info.ptr), num_samples, feature_dim, g_ptr, b_ptr, eps);
    py::array_t<float> result(x_info.shape);
    std::memcpy(result.request().ptr, y_vec.data(), y_vec.size() * sizeof(float));
    return result;
}

bool has_avx2_py() {
    return nv::has_avx2_hardware();
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "newton-vision C++ CNN & Transformer çekirdeği pybind11 modülü";

    m.def("has_cuda", &has_cuda_py, "CUDA GPU cihazı mevcut mu?");
    m.def("has_avx2", &has_avx2_py, "İşlemci AVX2 SIMD donanım ivmelendirmesini destekliyor mu?");

    m.def("conv2d", &conv2d_py,
          py::arg("x"), py::arg("w"), py::arg("bias") = py::none(),
          py::arg("stride") = 1, py::arg("padding") = 0,
          py::arg("dilation") = 1, py::arg("groups") = 1, py::arg("use_cuda") = false);


    m.def("conv2d_backward", &conv2d_backward_py,
          py::arg("grad_y"), py::arg("x"), py::arg("w"), py::arg("calc_bias") = true,
          py::arg("stride") = 1, py::arg("padding") = 0,
          py::arg("dilation") = 1, py::arg("groups") = 1);

    m.def("relu", &relu_py, py::arg("x"), py::arg("use_cuda") = false);
    m.def("relu_backward", &relu_backward_py, py::arg("grad_y"), py::arg("x"));

    m.def("leaky_relu", &leaky_relu_py, py::arg("x"), py::arg("negative_slope") = 0.01f);
    m.def("sigmoid", &sigmoid_py, py::arg("x"));
    m.def("tanh", &tanh_py, py::arg("x"));
    m.def("gelu", &gelu_py, py::arg("x"));

    m.def("maxpool2d", &maxpool2d_py, py::arg("x"), py::arg("kernel_size"), py::arg("stride") = py::none(), py::arg("use_cuda") = false);
    m.def("maxpool2d_backward", &maxpool2d_backward_py, py::arg("grad_y"), py::arg("x"), py::arg("kernel_size"), py::arg("stride") = py::none());

    m.def("avgpool2d", &avgpool2d_py, py::arg("x"), py::arg("kernel_size"), py::arg("stride") = py::none());
    m.def("batchnorm2d", &batchnorm2d_py, py::arg("x"), py::arg("gamma") = py::none(), py::arg("beta") = py::none(),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps") = 1e-5f);
    m.def("layernorm", &layernorm_py, py::arg("x"), py::arg("gamma") = py::none(), py::arg("beta") = py::none(), py::arg("eps") = 1e-5f);
}

