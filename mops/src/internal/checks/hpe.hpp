#ifndef MOPS_CHECKS_HPE_HPP
#define MOPS_CHECKS_HPE_HPP

#include "mops/tensor.hpp"
#include "utils.hpp"

template <typename scalar_t>
void check_hpe(
    mops::Tensor<scalar_t, 1> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A
) {
    check_sizes(A, "A", 0, output, "grad_output", 0, "hpe");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "hpe");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "hpe");
}

template <typename scalar_t>
void check_hpe_vjp(
    mops::Tensor<scalar_t, 2> grad_A,
    mops::Tensor<scalar_t, 1> grad_output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A
) {
    if (grad_A.data != nullptr) {
        check_sizes(grad_A, "grad_A", 0, A, "A", 0, "hpe_vjp");
        check_sizes(grad_A, "grad_A", 1, A, "A", 1, "hpe_vjp");
    }
    check_hpe(grad_output, A, C, indices_A);
}

#endif
