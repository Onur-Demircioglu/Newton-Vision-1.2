#pragma once
#include <cstdint>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cstring>
#include <memory>

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace nv {

enum class Device {
    CPU,
    CUDA
};

struct Shape {
    int n{1}, c{1}, h{1}, w{1};

    int64_t size() const {
        return static_cast<int64_t>(n) * c * h * w;
    }

    bool operator==(const Shape& other) const {
        return n == other.n && c == other.c && h == other.h && w == other.w;
    }
};

class Tensor {
private:
    Shape shape_;
    Device device_{Device::CPU};
    std::vector<float> cpu_buffer_;
    float* gpu_ptr_{nullptr};

public:
    Tensor() : shape_({0, 0, 0, 0}), device_(Device::CPU) {}

    Tensor(Shape shape, Device device = Device::CPU)
        : shape_(shape), device_(device) {
        allocate();
    }

    Tensor(Shape shape, const std::vector<float>& data, Device device = Device::CPU)
        : shape_(shape), device_(Device::CPU), cpu_buffer_(data) {
        if (static_cast<int64_t>(data.size()) != shape.size()) {
            throw std::invalid_argument("Tensor constructor: data size does not match shape size.");
        }
        if (device == Device::CUDA) {
            to(Device::CUDA);
        }
    }

    ~Tensor() {
        free_gpu();
    }

    // Copy Constructor
    Tensor(const Tensor& other)
        : shape_(other.shape_), device_(other.device_), cpu_buffer_(other.cpu_buffer_) {
        if (other.device_ == Device::CUDA) {
            allocate_gpu();
#ifdef HAS_CUDA
            cudaMemcpy(gpu_ptr_, other.gpu_ptr_, shape_.size() * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
        }
    }

    // Move Constructor
    Tensor(Tensor&& other) noexcept
        : shape_(other.shape_), device_(other.device_),
          cpu_buffer_(std::move(other.cpu_buffer_)), gpu_ptr_(other.gpu_ptr_) {
        other.gpu_ptr_ = nullptr;
        other.shape_ = {0, 0, 0, 0};
    }

    // Copy Assignment
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            free_gpu();
            shape_ = other.shape_;
            device_ = other.device_;
            cpu_buffer_ = other.cpu_buffer_;
            if (other.device_ == Device::CUDA) {
                allocate_gpu();
#ifdef HAS_CUDA
                cudaMemcpy(gpu_ptr_, other.gpu_ptr_, shape_.size() * sizeof(float), cudaMemcpyDeviceToDevice);
#endif
            }
        }
        return *this;
    }

    // Move Assignment
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            free_gpu();
            shape_ = other.shape_;
            device_ = other.device_;
            cpu_buffer_ = std::move(other.cpu_buffer_);
            gpu_ptr_ = other.gpu_ptr_;

            other.gpu_ptr_ = nullptr;
            other.shape_ = {0, 0, 0, 0};
        }
        return *this;
    }

    const Shape& shape() const { return shape_; }
    Device device() const { return device_; }
    int64_t size() const { return shape_.size(); }

    const float* data() const {
        if (device_ == Device::CUDA) {
            return gpu_ptr_;
        }
        return cpu_buffer_.data();
    }

    float* data() {
        if (device_ == Device::CUDA) {
            return gpu_ptr_;
        }
        return cpu_buffer_.data();
    }

    const std::vector<float>& cpu_data() const {
        return cpu_buffer_;
    }

    std::vector<float>& cpu_data() {
        return cpu_buffer_;
    }

    void to(Device target_device) {
        if (device_ == target_device) return;

        if (target_device == Device::CUDA) {
#ifdef HAS_CUDA
            allocate_gpu();
            cudaMemcpy(gpu_ptr_, cpu_buffer_.data(), shape_.size() * sizeof(float), cudaMemcpyHostToDevice);
            device_ = Device::CUDA;
#else
            throw std::runtime_error("CUDA acceleration is not compiled in this build of newton-vision.");
#endif
        } else {
            // Move from CUDA to CPU
#ifdef HAS_CUDA
            cpu_buffer_.resize(shape_.size());
            cudaMemcpy(cpu_buffer_.data(), gpu_ptr_, shape_.size() * sizeof(float), cudaMemcpyDeviceToHost);
            free_gpu();
            device_ = Device::CPU;
#endif
        }
    }

    int argmax() const {
        if (cpu_buffer_.empty()) {
            throw std::runtime_error("argmax requires CPU buffer data.");
        }
        int max_idx = 0;
        float max_val = cpu_buffer_[0];
        for (size_t i = 1; i < cpu_buffer_.size(); ++i) {
            if (cpu_buffer_[i] > max_val) {
                max_val = cpu_buffer_[i];
                max_idx = static_cast<int>(i);
            }
        }
        return max_idx;
    }

private:
    void allocate() {
        if (device_ == Device::CPU) {
            cpu_buffer_.resize(shape_.size(), 0.0f);
        } else {
            allocate_gpu();
        }
    }

    void allocate_gpu() {
#ifdef HAS_CUDA
        if (!gpu_ptr_ && shape_.size() > 0) {
            cudaError_t err = cudaMalloc(&gpu_ptr_, shape_.size() * sizeof(float));
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("CUDA Malloc failed: ") + cudaGetErrorString(err));
            }
        }
#endif
    }

    void free_gpu() {
#ifdef HAS_CUDA
        if (gpu_ptr_) {
            cudaFree(gpu_ptr_);
            gpu_ptr_ = nullptr;
        }
#endif
    }
};

} // namespace nv
