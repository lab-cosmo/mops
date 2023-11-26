#include <algorithm>
#include <stdexcept>
#include <string>

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

    // if (!std::is_sorted(indices_output.data, indices_output.data + indices_output.shape[0])) {
    //     throw std::runtime_error("`indices_output` values should be sorted");
    // }

    std::vector<std::vector<size_t>> write_list = get_write_list(indices_output);

    std::fill(output.data, output.data+output.shape[0]*output.shape[1]*output.shape[2], static_cast<scalar_t>(0.0));

    #pragma omp parallel for 
    for (size_t i_output=0; i_output<output.shape[0]; i_output++) {
        for (size_t index_inputs : write_list[i_output]) {
            for (size_t a_j=0; a_j<A.shape[1]; a_j++) {
                for (size_t b_j=0; b_j<B.shape[1]; b_j++) {
                    auto output_index = B.shape[1] * (A.shape[1] * i_output + a_j) + b_j;
                    output.data[output_index] += A.data[A.shape[1] * index_inputs + a_j]
                                            * B.data[B.shape[1] * index_inputs + b_j];
                }
            }
        }
    }
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
    // TODO: checks
    // TODO: set gradients to 0

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;

    if (calculate_grad_A || calculate_grad_B) {

        #pragma omp parallel for
        for (size_t i=0; i<A.shape[0]; i++) {
            int32_t i_output = indices_output.data[i];
            for (size_t a_j=0; a_j<A.shape[1]; a_j++) {
                auto grad_a_index = A.shape[1] * i + a_j;
                for (size_t b_j=0; b_j<B.shape[1]; b_j++) {
                    auto grad_b_index = B.shape[1] * i + b_j;
                    size_t output_index = B.shape[1] * (A.shape[1] * i_output + a_j) + b_j;
                    if (calculate_grad_A) grad_A.data[grad_a_index] += grad_output.data[output_index]
                        * B.data[B.shape[1] * i + b_j];
                    if (calculate_grad_B) grad_B.data[grad_b_index] += grad_output.data[output_index]
                        * A.data[A.shape[1] * i + a_j];
                }
            }
        }
    }

}
