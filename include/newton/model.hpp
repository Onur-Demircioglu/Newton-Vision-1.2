#pragma once
#include "newton/layers.hpp"
#include <map>
#include <string>
#include <fstream>
#include <iostream>

namespace nv {

class Model {
public:
    virtual ~Model() = default;
    virtual Tensor forward(const Tensor& input) = 0;

    virtual std::map<std::string, Tensor*>& state_dict() = 0;

    Tensor operator()(const Tensor& input) {
        return forward(input);
    }
};

// Pure C++ zero-dependency binary checkpoint serializer & deserializer
class BinaryModelSerializer {
public:
    static constexpr const char* MAGIC = "NVBIN100"; // 8 bytes magic header

    static void save_model_bin(Model& model, const std::string& filepath) {
        std::ofstream ofs(filepath, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filepath);
        }

        // 1. Write Header Magic
        ofs.write(MAGIC, 8);

        auto& params = model.state_dict();
        uint32_t tensor_count = static_cast<uint32_t>(params.size());
        ofs.write(reinterpret_cast<const char*>(&tensor_count), sizeof(uint32_t));

        for (const auto& [name, tensor_ptr] : params) {
            if (!tensor_ptr) continue;

            // Ensure tensor data is on CPU before saving
            Tensor cpu_tensor = *tensor_ptr;
            cpu_tensor.to(Device::CPU);

            // Write Name Length & Name
            uint32_t name_len = static_cast<uint32_t>(name.size());
            ofs.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
            ofs.write(name.data(), name_len);

            // Write Shape (4 ints)
            const Shape& shp = cpu_tensor.shape();
            int32_t dims[4] = { shp.n, shp.c, shp.h, shp.w };
            ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));

            // Write Total Float Count & Raw Float Data
            uint64_t float_count = static_cast<uint64_t>(cpu_tensor.size());
            ofs.write(reinterpret_cast<const char*>(&float_count), sizeof(uint64_t));
            ofs.write(reinterpret_cast<const char*>(cpu_tensor.data()), float_count * sizeof(float));
        }

        ofs.close();
    }

    static void load_model_bin(Model& model, const std::string& filepath) {
        std::ifstream ifs(filepath, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filepath);
        }

        // 1. Check Magic Header
        char magic_buf[9] = {0};
        ifs.read(magic_buf, 8);
        if (std::string(magic_buf, 8) != MAGIC) {
            throw std::runtime_error("Invalid binary checkpoint format in " + filepath + ". Expected header NVBIN100.");
        }

        uint32_t tensor_count = 0;
        ifs.read(reinterpret_cast<char*>(&tensor_count), sizeof(uint32_t));

        auto& params = model.state_dict();

        for (uint32_t i = 0; i < tensor_count; ++i) {
            uint32_t name_len = 0;
            ifs.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));

            std::string tensor_name(name_len, '\0');
            ifs.read(&tensor_name[0], name_len);

            int32_t dims[4] = {0};
            ifs.read(reinterpret_cast<char*>(dims), sizeof(dims));
            Shape loaded_shape{dims[0], dims[1], dims[2], dims[3]};

            uint64_t float_count = 0;
            ifs.read(reinterpret_cast<char*>(&float_count), sizeof(uint64_t));

            std::vector<float> loaded_data(float_count);
            ifs.read(reinterpret_cast<char*>(loaded_data.data()), float_count * sizeof(float));

            // Assign to model state_dict parameter if exists
            auto it = params.find(tensor_name);
            if (it != params.end() && it->second) {
                *(it->second) = Tensor(loaded_shape, loaded_data, Device::CPU);
            }
        }

        ifs.close();
    }
};

inline void save_model(Model& model, const std::string& filepath) {
    BinaryModelSerializer::save_model_bin(model, filepath);
}

inline void load_model(Model& model, const std::string& filepath) {
    BinaryModelSerializer::load_model_bin(model, filepath);
}

} // namespace nv
