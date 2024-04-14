#include <algorithm>
#include <vector>

#include "mops/opsa.hpp"

#include "internal/checks/opsa.hpp"
#include "internal/utils.hpp"

template <typename scalar_t>
void mops::outer_product_scatter_add(
    Tensor<scalar_t, 3> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    check_opsa(output, A, B, indices_output);

    size_t size_output = output.shape[0];
    size_t size_output_inner = output.shape[1] * output.shape[2];
    size_t size_a = A.shape[1];
    size_t size_b = B.shape[1];

    scalar_t *a_ptr = A.data;
    scalar_t *b_ptr = B.data;
    scalar_t *output_ptr = output.data;

    // For each index in the first dimension of the outputs,
    // get what indices in the inputs should write to it
    std::vector<std::vector<size_t>> write_list = get_write_list(indices_output);

    std::fill(
        output.data,
        output.data + output.shape[0] * output.shape[1] * output.shape[2],
        static_cast<scalar_t>(0.0)
    );

#pragma omp parallel for
    for (size_t i = 0; i < size_output; i++) {
        scalar_t *output_ptr_i = output_ptr + i * size_output_inner;
        // iterate over input indices that will write to the output index i
        for (size_t i_inputs : write_list[i]) {
            scalar_t *a_ptr_i_inputs = a_ptr + size_a * i_inputs;
            scalar_t *b_ptr_i_inputs = b_ptr + size_b * i_inputs;
            for (size_t a_j = 0; a_j < size_a; a_j++) {
                scalar_t *output_ptr_i_aj = output_ptr_i + a_j * size_b;
                scalar_t a_element = a_ptr_i_inputs[a_j];
                for (size_t b_j = 0; b_j < size_b; b_j++) {
                    output_ptr_i_aj[b_j] += a_element * b_ptr_i_inputs[b_j];
                }
            }
        }
    }
}

template <typename scalar_t>
void mops::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    check_opsa_vjp(grad_A, grad_B, grad_output, A, B, indices_output);

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;

    if (calculate_grad_A || calculate_grad_B) {
        size_t size_output_inner = grad_output.shape[1] * grad_output.shape[2];
        size_t size_ab = A.shape[0];
        size_t size_a = A.shape[1];
        size_t size_b = B.shape[1];

        if (calculate_grad_A) {
            std::fill(grad_A.data, grad_A.data + size_ab * size_a, static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_B) {
            std::fill(grad_B.data, grad_B.data + size_ab * size_b, static_cast<scalar_t>(0.0));
        }

        scalar_t *grad_a_ptr = grad_A.data;
        scalar_t *grad_b_ptr = grad_B.data;
        scalar_t *grad_output_ptr = grad_output.data;
        scalar_t *a_ptr = A.data;
        scalar_t *b_ptr = B.data;
        int32_t *indices_output_ptr = indices_output.data;

#pragma omp parallel for
        for (size_t i = 0; i < size_ab; i++) {
            scalar_t *grad_output_ptr_i =
                grad_output_ptr + indices_output_ptr[i] * size_output_inner;
            scalar_t *a_ptr_i = a_ptr + i * size_a;
            scalar_t *b_ptr_i = b_ptr + i * size_b;
            scalar_t *grad_a_ptr_i = grad_a_ptr + i * size_a;
            scalar_t *grad_b_ptr_i = grad_b_ptr + i * size_b;
            for (size_t a_j = 0; a_j < size_a; a_j++) {
                scalar_t *grad_output_ptr_i_aj = grad_output_ptr_i + a_j * size_b;
                scalar_t a_element = a_ptr_i[a_j];
                scalar_t *grad_a_element = grad_a_ptr_i + a_j;
                for (size_t b_j = 0; b_j < size_b; b_j++) {
                    if (calculate_grad_A) {
                        *grad_a_element += grad_output_ptr_i_aj[b_j] * b_ptr_i[b_j];
                    }
                    if (calculate_grad_B) {
                        grad_b_ptr_i[b_j] += grad_output_ptr_i_aj[b_j] * a_element;
                    }
                }
            }
        }
    }
}

template<typename scalar_t>
void mops::outer_product_scatter_add_vjp_vjp(
    Tensor<scalar_t, 3> /*grad_grad_output*/,
    Tensor<scalar_t, 2> /*grad_A_2*/,
    Tensor<scalar_t, 2> /*grad_B_2*/,
    Tensor<scalar_t, 2> /*grad_grad_A*/,
    Tensor<scalar_t, 2> /*grad_grad_B*/,
    Tensor<scalar_t, 3> /*grad_output*/,
    Tensor<scalar_t, 2> /*A*/,
    Tensor<scalar_t, 2> /*B*/,
    Tensor<int32_t, 1> /*indices_output*/
) {
    throw std::runtime_error("Not implemented yet.");
}
