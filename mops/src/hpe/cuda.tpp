#include <stdexcept>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 2> p
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
