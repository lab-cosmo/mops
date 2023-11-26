#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include "mops/hpe.hpp"
#include "mops/checks.hpp"
#include "mops/utils.hpp"


template<typename scalar_t, uint8_t polynomial_order>
void _homogeneous_polynomial_evaluation_templated_polynomial_order(
    mops::Tensor<scalar_t, 1> output,
    mops::Tensor<scalar_t, 2> A,
    mops::Tensor<scalar_t, 1> C,
    mops::Tensor<int32_t, 2> indices_A
) {

    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;

    size_t size_first_dimension = A.shape[0];
    size_t n_monomials = indices_A.shape[0];
    size_t n_possible_factors = A.shape[1];

    size_t size_second_dimension_a = A.shape[1];

    const size_t simd_element_count = get_simd_element_count<scalar_t>();

    size_t size_first_dimension_interleft = size_first_dimension / simd_element_count;
    size_t size_remainder = size_first_dimension % simd_element_count;

    scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
    scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
    interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);

    #pragma omp parallel for
    for (size_t i = 0; i < size_first_dimension_interleft; i++) {
        scalar_t* o_ptr_shifted = o_ptr + i * simd_element_count;
        std::array<scalar_t, simd_element_count> result = std::array<scalar_t, simd_element_count>();  // zero-initialized
        scalar_t* shifted_a_ptr = interleft_a_ptr + i*n_possible_factors*simd_element_count;
        int32_t* indices_A_ptr_row = indices_A_ptr;
        for (size_t j = 0; j < n_monomials; j++) {
            std::array<scalar_t, simd_element_count> temp;
            temp.fill(c_ptr[j]);
            for (size_t k = 0; k < polynomial_order; k++) {
                scalar_t* shifted_a_ptr_simd = shifted_a_ptr + indices_A_ptr_row[k]*simd_element_count;
                for (size_t l = 0; l < simd_element_count; l++) {
                    temp[l] *= shifted_a_ptr_simd[l];
                }
            }
            for (size_t l = 0; l < simd_element_count; l++) {
                result[l] += temp[l];
            }
            indices_A_ptr_row += polynomial_order;
        }
        for (size_t l = 0; l < simd_element_count; l++) {
            o_ptr_shifted[l] = result[l];
        }
    }

    std::vector<scalar_t> result = std::vector<scalar_t>(size_remainder, 0.0);
    scalar_t* shifted_a_ptr = remainder_a_ptr;
    scalar_t* o_ptr_shifted = o_ptr + size_first_dimension * simd_element_count;
    int32_t* indices_A_ptr_temp = indices_A_ptr;
    for (size_t j = 0; j < n_monomials; j++) {
        std::vector<scalar_t> temp = std::vector<scalar_t>(size_remainder, c_ptr[j]);
        for (size_t k = 0; k < polynomial_order; k++) {
            scalar_t* shifted_a_ptr_simd = shifted_a_ptr + indices_A_ptr_temp[k]*size_remainder;
            for (size_t l = 0; l < size_remainder; l++) {
                temp[l] *= shifted_a_ptr_simd[l];
            }
        }
        for (size_t l = 0; l < size_remainder; l++) {
            result[l] += temp[l];
        }
        indices_A_ptr_temp += polynomial_order;
    }
    for (size_t l = 0; l < size_remainder; l++) {
        o_ptr_shifted[l] = result[l];
    }

    delete[] interleft_a_ptr;
    delete[] remainder_a_ptr;
}


template<typename scalar_t>
void mops::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A
) {
    check_sizes(A, "A", 0, output, "O", 0, "hpe");
    check_sizes(C, "C", 0, indices_A,  "indices_A", 0, "hpe");
    check_index_tensor(indices_A,  "indices_A", A.shape[1], "hpe");

    size_t polynomial_order = indices_A.shape[1];

    if (polynomial_order <= 10) {
        const uint8_t polynomial_order_u8 = static_cast<uint8_t>(polynomial_order);
        switch (polynomial_order_u8) {
            case 0:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 0>(output, A, C, indices_A);
                return;
            case 1:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 1>(output, A, C, indices_A);
                return;
            case 2:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 2>(output, A, C, indices_A);
                return;
            case 3:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 3>(output, A, C, indices_A);
                return;
            case 4:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 4>(output, A, C, indices_A);
                return;
            case 5:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 5>(output, A, C, indices_A);
                return;
            case 6:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 6>(output, A, C, indices_A);
                return;
            case 7:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 7>(output, A, C, indices_A);
                return;
            case 8:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 8>(output, A, C, indices_A);
                return;
            case 9:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 9>(output, A, C, indices_A);
                return;
            case 10:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 10>(output, A, C, indices_A);
                return;
            default:
                break;
        }
    }

    throw std::runtime_error("Only up to polynomial order 10 is supported at the moment. Please contact the developers for more");
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
