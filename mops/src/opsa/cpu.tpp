#include <algorithm>
#include <stdexcept>
#include <string>
#include <execution>
#include <numeric>
#include <vector>

#include "mops/opsa.hpp"
#include "mops/checks.hpp"
#include "mops/utils.hpp"

#include <iostream>


template<typename scalar_t>
void mops::outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    check_sizes(A, "A", 0, B, "B", 0, "opsa");
    check_sizes(A, "A", 1, output, "output", 1, "opsa");
    check_sizes(B, "B", 1, output, "output", 2, "opsa");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsa");
    check_index_tensor(indices_output, "indices_output", output.shape[0], "opsa");

    if (!std::is_sorted(indices_output.data, indices_output.data + indices_output.shape[0])) {
        throw std::runtime_error("`indices_output` values should be sorted");
    }

    std::vector<int32_t> first_occurrences = find_first_occurrences(indices_output.data, indices_output.shape[0], output.shape[0]);

    std::fill(output.data, output.data+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));
    std::vector<size_t> indices(output.shape[0]);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t i_output) {
        int32_t index_start = first_occurrences[i_output];
        int32_t index_end = first_occurrences[i_output + 1];
        for (int32_t index_inputs = index_start; index_inputs < index_end; index_inputs++) {
            for (size_t a_j = 0; a_j < tensor_a.shape[1]; a_j++) {
                for (size_t b_j = 0; b_j < tensor_b.shape[1]; b_j++) {
                    auto output_index = tensor_b.shape[1] * (tensor_a.shape[1] * i_output + a_j) + b_j;
                    output.data[output_index] += tensor_a.data[tensor_a.shape[1] * index_inputs + a_j]
                                            * tensor_b.data[tensor_b.shape[1] * index_inputs + b_j];
                }
            }
        }
    });
}

template<typename scalar_t>
void mops::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {

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
