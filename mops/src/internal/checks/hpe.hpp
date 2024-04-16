#ifndef MOPS_CHECKS_HPE_HPP
#define MOPS_CHECKS_HPE_HPP

#include <string>
#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_hpe(
    mops::Tensor<scalar_t, 1> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A,
    std::string operation_name
) {
    check_sizes(A, "A", 0, output, "grad_output", 0, operation_name);
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, operation_name);
    if (operation_name.rfind("cuda_", 0) != 0) {
        // TODO: check CUDA index tensors
        check_index_tensor(indices_A, "indices_A", A.shape[1], operation_name);
    }
}

template <typename scalar_t>
void check_hpe_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 1> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A,
    std::string operation_name
) {
    if (grad_A.data != nullptr) {
        check_sizes(grad_A, "grad_A", 0, A, "A", 0, operation_name);
        check_sizes(grad_A, "grad_A", 1, A, "A", 1, operation_name);
    }
    check_hpe(grad_output, A, C, indices_A, operation_name);
}

template <typename scalar_t>
void check_hpe_vjp_vjp(
    mops::Tensor<scalar_t, 1> grad_grad_output,
    mops::Tensor<scalar_t, 2> grad_A_2,
    mops::Tensor<scalar_t, 2> grad_grad_A,
    mops::Tensor<scalar_t, 1> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A,
    std::string operation_name
) {
    if (grad_grad_output.data != nullptr) {
        check_sizes(
            grad_grad_output, "grad_grad_output", 0, grad_output, "grad_output", 0, operation_name
        );
    }
    if (grad_A_2.data != nullptr) {
        check_sizes(grad_A_2, "grad_A_2", 0, A, "A", 0, operation_name);
        check_sizes(grad_A_2, "grad_A_2", 1, A, "A", 1, operation_name);
    }
    if (grad_grad_A.data != nullptr) {
        check_sizes(grad_grad_A, "grad_grad_A", 0, A, "A", 0, operation_name);
        check_sizes(grad_grad_A, "grad_grad_A", 1, A, "A", 1, operation_name);
    }
    check_hpe_vjp(grad_grad_A, grad_output, A, C, indices_A, operation_name);
}

#endif
