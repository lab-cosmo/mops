#include <stdexcept>

#include "mops/hpe.hpp"

template<typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1>,
    Tensor<scalar_t, 2>,
    Tensor<scalar_t, 1>,
    Tensor<int32_t, 2>
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
