#include <stdexcept>

#include "mops/sap.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
