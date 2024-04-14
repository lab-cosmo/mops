#ifndef MOPS_CHECKS_SAP_HPP
#define MOPS_CHECKS_SAP_HPP

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
    mops::Tensor<int32_t, 1> indices_output
) {
    check_sizes(A, "A", 0, B, "B", 0, "sap");
    check_sizes(A, "A", 0, output, "output", 0, "sap");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "sap");
    check_sizes(C, "C", 0, indices_B, "indices_B", 0, "sap");
    check_sizes(C, "C", 0, indices_output, "indices_output", 0, "sap");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "sap");
    check_index_tensor(indices_B, "indices_B", B.shape[1], "sap");
    check_index_tensor(indices_output, "indices_output", output.shape[1], "sap");
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
    mops::Tensor<int32_t, 1> indices_output
) {
    if (grad_A.data != nullptr) {
        check_sizes(grad_A, "grad_A", 0, A, "A", 0, "sap_vjp");
        check_sizes(grad_A, "grad_A", 1, A, "A", 1, "sap_vjp");
    }
    if (grad_B.data != nullptr) {
        check_sizes(grad_B, "grad_B", 0, B, "B", 0, "sap_vjp");
        check_sizes(grad_B, "grad_B", 1, B, "B", 1, "sap_vjp");
    }
    check_sap(grad_output, A, B, C, indices_A, indices_B, indices_output);
}

#endif
