#include <stdexcept>

#include "mops/sap.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>,
    Tensor<int32_t, 1>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
