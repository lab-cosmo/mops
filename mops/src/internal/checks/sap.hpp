#ifndef MOPS_CHECKS_SAP_HPP
#define MOPS_CHECKS_SAP_HPP

#include <string>
#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_sap(
    mops::Tensor<scalar_t, 2> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 1> indices_A,
    mops::Tensor<int32_t, 1> indices_B,
    mops::Tensor<int32_t, 1> indices_output,
    std::string operation_name
) {
    check_sizes(A, "A", 0, B, "B", 0, operation_name);
    check_sizes(A, "A", 0, output, "output", 0, operation_name);
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, operation_name);
    check_sizes(C, "C", 0, indices_B, "indices_B", 0, operation_name);
    check_sizes(C, "C", 0, indices_output, "indices_output", 0, operation_name);
    if (operation_name.rfind("cuda_", 0) != 0) {
        check_index_tensor(indices_A, "indices_A", A.shape[1], operation_name);
        check_index_tensor(indices_B, "indices_B", B.shape[1], operation_name);
        check_index_tensor(indices_output, "indices_output", output.shape[1], operation_name);
    }
}

template <typename scalar_t>
void check_sap_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 2> grad_B,
    mops::Tensor<scalar_t, 2> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 1> indices_A,
    mops::Tensor<int32_t, 1> indices_B,
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
    check_sap(grad_output, A, B, C, indices_A, indices_B, indices_output, operation_name);
}

template <typename scalar_t>
void check_sap_vjp_vjp(
    mops::Tensor<scalar_t, 2> grad_grad_output,
    mops::Tensor<scalar_t, 2> grad_A_2,
    mops::Tensor<scalar_t, 2> grad_B_2,
    mops::Tensor<scalar_t, 2> grad_grad_A,
    mops::Tensor<scalar_t, 2> grad_grad_B,
    mops::Tensor<scalar_t, 2> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 1> indices_A,
    mops::Tensor<int32_t, 1> indices_B,
    mops::Tensor<int32_t, 1> indices_output,
    std::string operation_name
) {
    if (grad_grad_output.data != nullptr) {
        check_sizes(
            grad_grad_output, "grad_grad_output", 0, grad_output, "grad_output", 0, operation_name
        );
        check_sizes(
            grad_grad_output, "grad_grad_output", 1, grad_output, "grad_output", 1, operation_name
        );
    }
    if (grad_A_2.data != nullptr) {
        check_sizes(grad_A_2, "grad_A_2", 0, A, "A", 0, operation_name);
        check_sizes(grad_A_2, "grad_A_2", 1, A, "A", 1, operation_name);
    }
    if (grad_B_2.data != nullptr) {
        check_sizes(grad_B_2, "grad_B_2", 0, B, "B", 0, operation_name);
        check_sizes(grad_B_2, "grad_B_2", 1, B, "B", 1, operation_name);
    }
    if (grad_grad_A.data != nullptr) {
        check_sizes(grad_grad_A, "grad_grad_A", 0, A, "A", 0, operation_name);
        check_sizes(grad_grad_A, "grad_grad_A", 1, A, "A", 1, operation_name);
    }
    if (grad_grad_B.data != nullptr) {
        check_sizes(grad_grad_B, "grad_grad_B", 0, B, "B", 0, operation_name);
        check_sizes(grad_grad_B, "grad_grad_B", 1, B, "B", 1, operation_name);
    }
    check_sap_vjp(
        grad_grad_A, grad_grad_B, grad_output, A, B, C, indices_A, indices_B, indices_output, operation_name
    );
}

#endif
