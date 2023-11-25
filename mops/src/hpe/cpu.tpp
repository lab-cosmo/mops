#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <iostream>

#include "mops/hpe.hpp"
#include "mops/checks.hpp"

template<typename scalar_t>
void mops::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
) {
    check_sizes(A, "A", 0, output, "output", 0, "hpe");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "hpe");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "hpe");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;

    size_t size_batch_dimension = A.shape[0];
    size_t n_monomials = indices_A.shape[0];
    size_t polynomial_order = indices_A.shape[1];
    size_t n_possible_factors = A.shape[1];

    for (size_t i = 0; i < size_batch_dimension; i++) {
        scalar_t result = 0.0;
        scalar_t* shifted_a_ptr = a_ptr + i*n_possible_factors;
        int32_t* indices_A_ptr_row = indices_A_ptr;
        for (size_t j = 0; j < n_monomials; j++) {
            scalar_t temp = c_ptr[j];
            for (uint8_t k = 0; k < polynomial_order; k++) {
                temp *= shifted_a_ptr[indices_A_ptr_row[k]];
            }
            result += temp;
            indices_A_ptr_row += polynomial_order;
        }
        o_ptr[i] = result;
    }

}


template<typename scalar_t>
void mops::homogeneous_polynomial_evaluation_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 1> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
) {
    // TODO: check everything

    if (grad_A.data != nullptr) {
        
        scalar_t* grad_o_ptr = grad_output.data;
        scalar_t* a_ptr = A.data;
        scalar_t* c_ptr = C.data;
        int32_t* indices_A_ptr = indices_A.data;
        scalar_t* grad_A_ptr = grad_A.data;

        size_t size_batch_dimension = A.shape[0];
        size_t n_monomials = indices_A.shape[0];
        size_t polynomial_order = indices_A.shape[1];
        size_t n_possible_factors = A.shape[1];

        for (size_t i = 0; i < size_batch_dimension; i++) {
            scalar_t grad_output_i = grad_o_ptr[i];
            size_t i_shift = i * n_possible_factors;
            scalar_t* a_ptr_i = a_ptr + i_shift;
            scalar_t* grad_A_ptr_i = grad_A_ptr + i_shift;
            int32_t* indices_ptr_j = indices_A_ptr;
            for (size_t j = 0; j < n_monomials; j++) {
                scalar_t base_multiplier = grad_output_i*c_ptr[j];
                for (uint8_t i_factor = 0; i_factor < polynomial_order; i_factor++) {
                    scalar_t temp = base_multiplier;
                    for (uint8_t j_factor = 0; j_factor < polynomial_order; j_factor++) {
                        if (j_factor == i_factor) continue;
                        temp *= a_ptr_i[indices_ptr_j[j_factor]];
                    }
                    grad_A_ptr_i[indices_ptr_j[i_factor]] += temp;
                }
                indices_ptr_j += polynomial_order;
            }
        }

    }

}
