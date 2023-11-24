#include <torch/script.h>
#include <vector>
#include <string>

#include "mops/torch/utils.hpp" 


void check_all_same_device(std::vector<torch::Tensor> tensors) {
    if (tensors.size() == 0) return;
    auto device = tensors[0].device();
    for (auto tensor : tensors) {
        if (tensor.device() != device) C10_THROW_ERROR(
            ValueError,
            "All tensors must be on the same device, found" + tensor.device().str() + " and " + device.str()
        );
    }
}

void check_all_same_dtype(std::vector<torch::Tensor> tensors) {
    if (tensors.size() == 0) return;
    auto dtype = tensors[0].dtype();
    for (auto tensor : tensors) {
        if (tensor.dtype() != dtype) C10_THROW_ERROR(
            TypeError,
            "All tensors must be of the same dtype, found" + std::string(tensor.dtype().name()) + " and " + std::string(dtype.name())
        );
    }
}

void check_number_of_dimensions(torch::Tensor tensor, int64_t expected, std::string tensor_name, std::string operation_name) {
    if (tensor.dim() != expected) C10_THROW_ERROR(
        ValueError,
        "Expected " + tensor_name + " to have " + std::to_string(expected) + " dimensions, in " + operation_name + ", found" + std::to_string(tensor.dim())
    );
}
