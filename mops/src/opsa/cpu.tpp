#include <cassert>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add(
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
    if (tensor_a_shape_1 != tensor_b_shape_1) {
        throw std::runtime_error(
            "A and B tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(tensor_a_shape_1) + " and " +
            std::to_string(tensor_b_shape_1)
        );
    }

    if (tensor_a_shape_1 != indexes_shape_1) {
        throw std::runtime_error(
            "indexes must contain the same number of elements as the first "
            "dimension of A and B , got " + std::to_string(indexes_shape_1) +
            " and " + std::to_string(tensor_a_shape_1)
        );
    }

    if (tensor_a_shape_2 * tensor_b_shape_2 != output_shape_2) {
        throw std::runtime_error(
            "output tensor must have space for " + std::to_string(tensor_a_shape_2 * tensor_b_shape_2) +
            " along the second dimension, got " + std::to_string(output_shape_2)
        );
    }

    for (size_t i=0; i<tensor_a_shape_1; i++) {
        auto i_output = indexes[i];
        assert(i_output < output_shape_1);
        for (size_t a_j=0; a_j<tensor_a_shape_2; a_j++) {
            for (size_t b_j=0; b_j<tensor_b_shape_2; b_j++) {
                auto output_index = tensor_b_shape_2 * (tensor_a_shape_2 * i_output + a_j) + b_j;
                output[output_index] += tensor_a[tensor_a_shape_2 * i + a_j]
                                      * tensor_b[tensor_b_shape_2 * i + b_j];
            }
        }
    }
}
