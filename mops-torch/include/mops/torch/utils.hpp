#include <torch/script.h>

#include "mops/tensor.hpp"

namespace mops_torch::details {

template <typename T> static mops::Tensor<T, 1> torch_to_mops_1d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 1);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0))},
    };
}

template <typename T> static mops::Tensor<T, 2> torch_to_mops_2d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 2);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0)), static_cast<size_t>(tensor.size(1))},
    };
}

template <typename T> static mops::Tensor<T, 3> torch_to_mops_3d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 3);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0)),
         static_cast<size_t>(tensor.size(1)),
         static_cast<size_t>(tensor.size(2))},
    };
}

void check_all_same_device(const std::vector<torch::Tensor> &tensors);

void check_floating_dtype(const std::vector<torch::Tensor> &tensors);

void check_integer_dtype(const std::vector<torch::Tensor> &tensors);

void check_all_same_dtype(std::vector<torch::Tensor> tensors);

void check_number_of_dimensions(
    torch::Tensor tensor, int64_t expected, std::string tensor_name, std::string operation_name
);

} // namespace mops_torch::details
