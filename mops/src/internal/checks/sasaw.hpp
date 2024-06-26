#ifndef MOPS_CHECKS_SASAW_HPP
#define MOPS_CHECKS_SASAW_HPP

#include <string>
#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_sasaw(
    mops::Tensor<scalar_t, 3> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<scalar_t, 3> W,
    mops::Tensor<int32_t, 1> indices_A,
    mops::Tensor<int32_t, 1> indices_W_1,
    mops::Tensor<int32_t, 1> indices_W_2,
    mops::Tensor<int32_t, 1> indices_output_1,
    mops::Tensor<int32_t, 1> indices_output_2,
    std::string operation_name
) {
    check_sizes(A, "A", 0, B, "B", 0, operation_name);
    check_sizes(W, "W", 0, output, "output", 0, operation_name);
    check_sizes(B, "B", 1, W, "W", 2, operation_name);
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, operation_name);
    check_sizes(C, "C", 0, indices_W_2, "indices_W_2", 0, operation_name);
    check_sizes(C, "C", 0, indices_output_2, "indices_output_2", 0, operation_name);
    check_sizes(A, "A", 0, indices_output_1, "indices_output_1", 0, operation_name);
    check_sizes(A, "A", 0, indices_W_1, "indices_W_1", 0, operation_name);
    if (operation_name.rfind("cuda_", 0) != 0) {
        check_index_tensor(indices_A, "indices_A", A.shape[1], operation_name);
        check_index_tensor(indices_W_1, "indices_W_1", output.shape[0], operation_name);
        check_index_tensor(indices_W_2, "indices_W_2", B.shape[1], operation_name);
        check_index_tensor(indices_output_1, "indices_output_1", output.shape[0], operation_name);
        check_index_tensor(indices_output_2, "indices_output_2", output.shape[1], operation_name);
    }
}

template <typename scalar_t>
void check_sasaw_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 2> grad_B,
    mops::Tensor<scalar_t, 3> grad_W,
    mops::Tensor<scalar_t, 3> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<scalar_t, 3> W,
    mops::Tensor<int32_t, 1> indices_A,
    mops::Tensor<int32_t, 1> indices_W_1,
    mops::Tensor<int32_t, 1> indices_W_2,
    mops::Tensor<int32_t, 1> indices_output_1,
    mops::Tensor<int32_t, 1> indices_output_2,
    std::string operation_name
) {
    if (grad_A.data != nullptr) {
        check_sizes(grad_A, "grad_A", 0, A, "A", 0, operation_name);
        check_sizes(grad_A, "grad_A", 1, A, "A", 1, operation_name);
    }
    if (grad_B.data != nullptr) {
        check_sizes(grad_B, "grad_B", 0, B, "B", 0, operation_name);
        check_sizes(grad_B, "grad_B", 1, B, "B", 1, operation_name);
    }
    if (grad_W.data != nullptr) {
        check_sizes(grad_W, "grad_W", 0, W, "W", 0, operation_name);
        check_sizes(grad_W, "grad_W", 1, W, "W", 1, operation_name);
        check_sizes(grad_W, "grad_W", 2, W, "W", 2, operation_name);
    }
    check_sasaw(
        grad_output,
        A,
        B,
        C,
        W,
        indices_A,
        indices_W_1,
        indices_W_2,
        indices_output_1,
        indices_output_2,
        operation_name
    );
}

#endif
