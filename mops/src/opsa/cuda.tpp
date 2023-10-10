#include <stdexcept>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::cuda::outer_product_scatter_add(
    scalar_t* output,
    size_t output_shape_1,
    size_t output_shape_2,
    const scalar_t* tensor_a,
    size_t tensor_a_shape_1,
    size_t tensor_a_shape_2,
    const scalar_t* tensor_b,
    size_t tensor_b_shape_1,
    size_t tensor_b_shape_2,
    const int32_t* indexes,
    size_t indexes_shape_1
) {
    throw std::runtime_error("CUDA implementation does not exist yet");
}
