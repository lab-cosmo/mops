#ifndef MOPS_CHECKS_OPSA_HPP
#define MOPS_CHECKS_OPSA_HPP

#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_opsa(
    mops::Tensor<scalar_t, 3> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<int32_t, 1> indices_output
) {
    check_sizes(A, "A", 0, B, "B", 0, "opsa");
    check_sizes(A, "A", 1, output, "output", 1, "opsa");
    check_sizes(B, "B", 1, output, "output", 2, "opsa");
    check_sizes(A, "A", 0, indices_output, "indices_output", 0, "opsa");
    check_index_tensor(indices_output, "indices_output", output.shape[0], "opsa");
}

template <typename scalar_t>
void check_opsa_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 2> grad_B,
    mops::Tensor<scalar_t, 3> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 2> B,
    mops::Tensor<int32_t, 1> indices_output
) {
    if (grad_A.data != nullptr) {
        check_sizes(grad_A, "grad_A", 0, A, "A", 0, "opsa_vjp");
        check_sizes(grad_A, "grad_A", 1, A, "A", 1, "opsa_vjp");
    }
    if (grad_B.data != nullptr) {
        check_sizes(grad_B, "grad_B", 0, B, "B", 0, "opsa_vjp");
        check_sizes(grad_B, "grad_B", 1, B, "B", 1, "opsa_vjp");
    }
    check_opsa(grad_output, A, B, indices_output);
}

#endif
