#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/hpe.hpp"
#include "mops/checks.hpp"

template<typename scalar_t>
void mops::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 2> p
) {
    check_sizes(tensor_a, "A", 0, output, "O", 0, "hpe");
    check_sizes(tensor_c, "C", 0, p, "P", 0, "hpe");
    check_index_tensor(p, "P", tensor_a.shape[1], "hpe");

    scalar_t* o_ptr = output.data;
    scalar_t* a_ptr = tensor_a.data;
    scalar_t* c_ptr = tensor_c.data;
    int32_t* p_ptr = p.data;

    size_t size_batch_dimension = tensor_a.shape[0];
    size_t n_monomials = p.shape[0];
    size_t polynomial_order = p.shape[1];
    size_t n_possible_factors = tensor_a.shape[1];

    for (size_t i = 0; i < size_batch_dimension; i++) {
        scalar_t result = 0.0;
        scalar_t* shifted_a_ptr = a_ptr + i*n_possible_factors;
        int32_t* p_ptr_temp = p_ptr;
        for (size_t j = 0; j < n_monomials; j++) {
            scalar_t temp = c_ptr[j];
            for (size_t k = 0; k < polynomial_order; k++) {
                temp *= shifted_a_ptr[p_ptr_temp[k]];
            }
            result += temp;
            p_ptr_temp += polynomial_order;
        }
        o_ptr[i] = result;
    }

}
