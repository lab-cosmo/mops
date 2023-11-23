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
    check_sizes(tensor_a, "A", 0, output, "O", 0, "hpe");
    check_sizes(tensor_c, "C", 0, p, "P", 0, "hpe");
    check_index_tensor(p, "P", tensor_a.shape[1], "hpe");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = A.data;
    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;

    long size_batch_dimension = A.shape[0];
    long n_monomials = indices_A.shape[0];
    long polynomial_order = indices_A.shape[1];
    long n_possible_factors = A.shape[1];

    for (size_t i = 0; i < size_batch_dimension; i++) {
        scalar_t result = 0.0;
        scalar_t* shifted_a_ptr = a_ptr + i*n_possible_factors;
        int32_t* indices_A_ptr_temp = indices_A_ptr;
        for (long j = 0; j < n_monomials; j++) {
            scalar_t temp = c_ptr[j];
            for (uint8_t k = 0; k < polynomial_order; k++) {
                temp *= shifted_a_ptr[indices_A_ptr_temp[k]];
            }
            result += temp;
            indices_A_ptr_temp += indices_Aolynomial_order;
        }
        o_ptr[i] = result;
    }

}
