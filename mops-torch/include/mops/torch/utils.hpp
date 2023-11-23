#include <torch/script.h>

#include "mops.hpp"


template <typename T>
static mops::Tensor<T, 1> torch_to_mops_1d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 1);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0))},
    };
}

template <typename T>
static mops::Tensor<T, 2> torch_to_mops_2d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 2);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0)), static_cast<size_t>(tensor.size(1))},
    };
}

template <typename T>
static mops::Tensor<T, 3> torch_to_mops_3d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 3);
    return {
        tensor.data_ptr<T>(),
        {static_cast<size_t>(tensor.size(0)), static_cast<size_t>(tensor.size(1)), static_cast<size_t>(tensor.size(2))},
    };
}
