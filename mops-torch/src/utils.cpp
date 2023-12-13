#include <string>
#include <torch/script.h>
#include <vector>

#include "mops/torch/utils.hpp"

void mops_torch::details::check_all_same_device(const std::vector<torch::Tensor> &tensors) {
    if (tensors.empty()) {
        return;
    }
    auto device = tensors[0].device();
    for (const auto &tensor : tensors) {
        if (tensor.device() != device) {
            C10_THROW_ERROR(
                ValueError,
                "All tensors must be on the same device, found" + tensor.device().str() + " and " +
                    device.str()
            );
        }
    }
}

void mops_torch::details::check_floating_dtype(const std::vector<torch::Tensor> &tensors) {
    if (tensors.empty()) {
        return;
    }
    auto dtype = tensors[0].dtype();
    if ((dtype != torch::kF64) && (dtype != torch::kF32)) {
        C10_THROW_ERROR(
            TypeError,
            "Found dtype" + std::string(dtype.name()) + ", only float32 and float64 are supported"
        );
    }
    for (const auto &tensor : tensors) {
        if (tensor.dtype() != dtype) {
            C10_THROW_ERROR(
                TypeError,
                "All floating point tensors must be of the same dtype, found " +
                    std::string(tensor.dtype().name()) + " and " + std::string(dtype.name())
            );
        }
    }
}

void mops_torch::details::check_integer_dtype(const std::vector<torch::Tensor> &tensors) {
    for (const auto &tensor : tensors) {
        if (tensor.dtype() != torch::kI32) {
            C10_THROW_ERROR(
                TypeError,
                "All index tensors must be of type int32, found " + std::string(tensor.dtype().name())
            );
        }
    }
}

void mops_torch::details::check_number_of_dimensions(
    torch::Tensor tensor, int64_t expected, std::string tensor_name, std::string operation_name
) {
    if (tensor.dim() != expected) {
        C10_THROW_ERROR(
            ValueError,
            "Expected " + tensor_name + " to have " + std::to_string(expected) +
                " dimensions, in " + operation_name + ", found" + std::to_string(tensor.dim())
        );
    }
}
