#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"
#include "mops/checks.hpp"

using namespace mops;

template<typename scalar_t>
static void check_inputs_shape(
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    // check_sizes(tensor_a, "A", 0, tensor_b, "B", 0, "opsa");
    // check_sizes(tensor_a, "A", 1, output, "O", 1, "opsa");
    // check_sizes(tensor_b, "B", 1, output, "O", 2, "opsa");
    // check_sizes(tensor_a, "A", 0, indexes, "P", 0, "opsa");
    // check_index_tensor(indexes, "P", output.shape[0], "opsa");

    if (!std::is_sorted(indices_output.data, indices_output.data + indices_output.shape[0])) {
        throw std::runtime_error("`indices_output` values should be sorted");
    }

    for (size_t i=0; i<A.shape[0]; i++) {
        auto i_output = indices_output.data[i];
        assert(i_output < output.shape[0]);
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
            assert(i_output < grad_output.shape[0]);
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
            assert(i_output < grad_output.shape[0]);
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
