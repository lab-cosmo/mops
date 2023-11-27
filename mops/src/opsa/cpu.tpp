#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"

using namespace mops;

template<typename scalar_t>
static void check_inputs_shape(
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    if (A.shape[0] != B.shape[0]) {
        throw std::runtime_error(
            "A and B tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(A.shape[0]) + " and " +
            std::to_string(B.shape[0])
        );
    }

    if (A.shape[0] != indices_output.shape[0]) {
        throw std::runtime_error(
            "indices_output must contain the same number of elements as the first "
            "dimension of A and B , got " + std::to_string(indices_output.shape[0]) +
            " and " + std::to_string(A.shape[0])
        );
    }
}

template<typename scalar_t>
void mops::outer_product_scatter_add(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    check_inputs_shape(A, B, indices_output);

    if (A.shape[1] * B.shape[1] != output.shape[1]) {
        throw std::runtime_error(
            "output tensor must have space for " + std::to_string(A.shape[1] * B.shape[1]) +
            " along the second dimension, got " + std::to_string(output.shape[1])
        );
    }

    if (!std::is_sorted(indices_output.data, indices_output.data + indices_output.shape[0])) {
        throw std::runtime_error("`indices_output` values should be sorted");
    }

    std::fill(output.data, output.data+output.shape[0]*output.shape[1], static_cast<scalar_t>(0.0));

    for (size_t i=0; i<A.shape[0]; i++) {
        auto i_output = indices_output.data[i];
        for (size_t a_j=0; a_j<A.shape[1]; a_j++) {
            for (size_t b_j=0; b_j<B.shape[1]; b_j++) {
                auto output_index = B.shape[1] * (A.shape[1] * i_output + a_j) + b_j;
                output.data[output_index] += A.data[A.shape[1] * i + a_j]
                                           * B.data[B.shape[1] * i + b_j];
            }
        }
    }
}

template<typename scalar_t>
void mops::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    check_inputs_shape(A, B, indices_output);
    if (A.shape[1] * B.shape[1] != grad_output.shape[1]) {
        throw std::runtime_error(
            "`grad_output` tensor must have " + std::to_string(A.shape[1] * B.shape[1]) +
            " elements in the second dimension, got " + std::to_string(grad_output.shape[1])
        );
    }

    if (grad_A.data != nullptr) {
        if (A.shape[0] != grad_A.shape[0] || A.shape[1] != grad_A.shape[1]) {
            throw std::runtime_error(
                "A and grad_A tensors must have the same shape"
            );
        }

        for (size_t i=0; i<A.shape[0]; i++) {
            auto i_output = indices_output.data[i];
            for (size_t a_j=0; a_j<A.shape[1]; a_j++) {
                auto grad_index = A.shape[1] * i + a_j;
                auto sum = 0.0;
                for (size_t b_j=0; b_j<B.shape[1]; b_j++) {
                    auto output_index = B.shape[1] * (A.shape[1] * i_output + a_j) + b_j;
                    sum += grad_output.data[output_index]
                         * B.data[B.shape[1] * i + b_j];
                }
                grad_A.data[grad_index] += sum;
            }
        }
    }

    if (grad_B.data != nullptr) {
        if (B.shape[0] != grad_B.shape[0] || B.shape[1] != grad_B.shape[1]) {
            throw std::runtime_error(
                "B and grad_B tensors must have the same shape"
            );
        }

        for (size_t i=0; i<A.shape[0]; i++) {
            auto i_output = indices_output.data[i];
            for (size_t b_j=0; b_j<B.shape[1]; b_j++) {
                auto grad_index = B.shape[1] * i + b_j;
                auto sum = 0.0;
                for (size_t a_j=0; a_j<A.shape[1]; a_j++) {
                    auto output_index = B.shape[1] * (A.shape[1] * i_output + a_j) + b_j;
                    sum += grad_output.data[output_index]
                         * A.data[A.shape[1] * i + a_j];
                }
                grad_B.data[grad_index] += sum;
            }
        }
    }
}
