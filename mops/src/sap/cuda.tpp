#include <stdexcept>

#include "mops/sap.hpp"

template<typename scalar_t>
void mops::cuda::sparse_accumulation_of_products(
    [[maybe_unused]] Tensor<scalar_t, 2> output,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_a,
    [[maybe_unused]] Tensor<scalar_t, 2> tensor_b,
    [[maybe_unused]] Tensor<scalar_t, 1> tensor_c,
    [[maybe_unused]] Tensor<int32_t, 1> p_a,
    [[maybe_unused]] Tensor<int32_t, 1> p_b,
    [[maybe_unused]] Tensor<int32_t, 1> p_o
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
