#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

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
        int32_t* indices_A_ptr_temp = indices_A_ptr;
        for (size_t j = 0; j < n_monomials; j++) {
            scalar_t temp = c_ptr[j];
            for (uint8_t k = 0; k < polynomial_order; k++) {
                temp *= shifted_a_ptr[indices_A_ptr_temp[k]];
            }
            result += temp;
            indices_A_ptr_temp += polynomial_order;
        }
        o_ptr[i] = result;
    }

}
