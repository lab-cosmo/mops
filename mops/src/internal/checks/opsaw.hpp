#ifndef MOPS_CHECKS_OPSAW_HPP
#define MOPS_CHECKS_OPSAW_HPP

#include <string>
#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_opsaw(
    mops::Tensor<scalar_t, 3> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 2> W,
    mops::Tensor<int32_t, 1> indices_W,
    mops::Tensor<int32_t, 1> indices_output,
    std::string operation_name
) {
    check_sizes(A, "A", 0, B, "B", 0, operation_name);
    check_sizes(A, "A", 1, output, "output", 1, operation_name);
    check_sizes(B, "B", 1, output, "output", 2, operation_name);
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, operation_name);
    check_sizes(A, "A", 0, indices_W, "indices_W", 0, operation_name);
    check_sizes(W, "W", 0, output, "output", 0, operation_name);
    check_sizes(B, "B", 1, W, "W", 1, operation_name);
    if (operation_name.rfind("cuda_", 0) != 0) {
        check_index_tensor(indices_output, "indices_output", output.shape[0], operation_name);
        check_index_tensor(indices_W, "indices_W", output.shape[0], operation_name);
    }
}

template <typename scalar_t>
void check_opsaw_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 2> grad_B,
    mops::Tensor<scalar_t, 2> grad_W,
    mops::Tensor<scalar_t, 3> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 2> W,
    mops::Tensor<int32_t, 1> indices_W,
    mops::Tensor<int32_t, 1> indices_output,
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
    }
    check_opsaw(grad_output, A, B, W, indices_W, indices_output, operation_name);
}

#endif
