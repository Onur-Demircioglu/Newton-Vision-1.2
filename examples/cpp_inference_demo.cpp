#include <iostream>
#include <chrono>
#include <vector>
#include "newton/newton.hpp"

class CppLeNet : public nv::Model {
public:
    nv::Conv2d conv1{1, 6, 5, 1, 2};
    nv::ReLU relu1;
    nv::MaxPool2d pool1{2, 2};

    nv::Conv2d conv2{6, 16, 5};
    nv::ReLU relu2;
    nv::MaxPool2d pool2{2, 2};

    nv::Linear fc1{16 * 5 * 5, 120};
    nv::ReLU relu3;
    nv::Linear fc2{120, 84};
    nv::ReLU relu4;
    nv::Linear fc3{84, 10};

    std::map<std::string, nv::Tensor*> params;

    CppLeNet() {
        params["conv1.weight"] = &conv1.weight;
        params["conv1.bias"] = &conv1.bias;
        params["conv2.weight"] = &conv2.weight;
        params["conv2.bias"] = &conv2.bias;
        params["fc1.weight"] = &fc1.weight;
        params["fc1.bias"] = &fc1.bias;
        params["fc2.weight"] = &fc2.weight;
        params["fc2.bias"] = &fc2.bias;
        params["fc3.weight"] = &fc3.weight;
        params["fc3.bias"] = &fc3.bias;
    }

    std::map<std::string, nv::Tensor*>& state_dict() override {
        return params;
    }

    nv::Tensor forward(const nv::Tensor& x) override {
        auto h = pool1(relu1(conv1(x)));
        h = pool2(relu2(conv2(h)));
        
        // Flatten 4D tensor to 2D matrix shape for Linear layer
        const auto& s = h.shape();
        nv::Tensor flat(nv::Shape{s.n, s.c * s.h * s.w, 1, 1}, h.cpu_data(), h.device());
        
        h = relu3(fc1(flat));
        h = relu4(fc2(h));
        return fc3(h);
    }
};

int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << " 🌌 newton-vision v2.0: PURE C++ STANDALONE INFERENCE ENGINE" << std::endl;
    std::cout << "================================================================" << std::endl;

    bool cuda = nv::is_cuda_available();
    std::cout << "  • CUDA Acceleration Available: " << (cuda ? "YES (GPU)" : "NO (OpenMP CPU)") << std::endl;

    CppLeNet model;

    // Create synthetic input tensor (1, 1, 28, 28)
    nv::Shape input_shape{1, 1, 28, 28};
    std::vector<float> dummy_data(1 * 1 * 28 * 28, 0.5f);
    nv::Tensor input(input_shape, dummy_data, cuda ? nv::Device::CUDA : nv::Device::CPU);

    // Warmup pass
    auto logits = model(input);

    // Save & reload binary checkpoint test in C++
    try {
        nv::save_model(model, "cpp_lenet_checkpoint.nv.bin");
        std::cout << "  ✓ Pure C++ Binary Checkpoint Saved: cpp_lenet_checkpoint.nv.bin" << std::endl;
        
        nv::load_model(model, "cpp_lenet_checkpoint.nv.bin");
        std::cout << "  ✓ Pure C++ Binary Checkpoint Loaded: cpp_lenet_checkpoint.nv.bin" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "  ❌ Checkpoint Error: " << e.what() << std::endl;
    }

    // Benchmarking 100 inference iterations in pure C++
    int iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        logits = model(input);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_latency = total_ms / iterations;
    double fps = 1000.0 / avg_latency;

    if (logits.device() == nv::Device::CUDA) {
        logits.to(nv::Device::CPU);
    }

    std::cout << "\n🚀 PURE C++ INFERENCE ENGINE PERFORMANCE:" << std::endl;
    std::cout << "  • Input Shape          : (1, 1, 28, 28)" << std::endl;
    std::cout << "  • Output Logits Shape  : (" << logits.shape().n << ", " << logits.shape().c << ")" << std::endl;
    std::cout << "  • Predicted Top Class  : " << logits.argmax() << std::endl;
    std::cout << "  • Average Latency      : " << avg_latency << " ms per frame" << std::endl;
    std::cout << "  • High-Speed Throughput: " << fps << " FPS" << std::endl;
    std::cout << "================================================================" << std::endl;

    return 0;
}
