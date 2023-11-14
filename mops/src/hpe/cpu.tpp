#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include "mops/hpe.hpp"
#include "mops/utils.hpp"


template<typename scalar_t, uint8_t polynomial_order>
void _homogeneous_polynomial_evaluation_templated_polynomial_order(
    mops::Tensor<scalar_t, 1> output,
    mops::Tensor<scalar_t, 2> tensor_a,
    mops::Tensor<scalar_t, 1> tensor_c,
    mops::Tensor<int32_t, 2> p
) {

    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = tensor_c.data;
    int32_t* p_ptr = p.data;

    size_t size_first_dimension = tensor_a.shape[0];
    size_t n_monomials = p.shape[0];
    size_t n_possible_factors = tensor_a.shape[1];

    size_t size_second_dimension_a = tensor_a.shape[1];

    const size_t simd_element_count = get_simd_element_count<scalar_t>();

    size_t size_first_dimension_interleft = size_first_dimension / simd_element_count;
    size_t size_remainder = size_first_dimension % simd_element_count;

    scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
    scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
    interleave_tensor<scalar_t, simd_element_count>(tensor_a, interleft_a_ptr, remainder_a_ptr);

    scalar_t* o_ptr_shifted = o_ptr;
    for (size_t i = 0; i < size_first_dimension_interleft; i++) {
        std::array<scalar_t, simd_element_count> result = std::array<scalar_t, simd_element_count>();  // zero-initialized
        scalar_t* shifted_a_ptr = interleft_a_ptr + i*n_possible_factors*simd_element_count;
        int32_t* p_ptr_temp = p_ptr;
        for (size_t j = 0; j < n_monomials; j++) {
            std::array<scalar_t, simd_element_count> temp;
            temp.fill(c_ptr[j]);
            for (size_t k = 0; k < polynomial_order; k++) {
                scalar_t* shifted_a_ptr_simd = shifted_a_ptr + p_ptr_temp[k]*simd_element_count;
                for (size_t l = 0; l < simd_element_count; l++) {
                    temp[l] *= shifted_a_ptr_simd[l];
                }
            }
            for (size_t l = 0; l < simd_element_count; l++) {
                result[l] += temp[l];
            }
            p_ptr_temp += polynomial_order;
        }
        for (size_t l = 0; l < simd_element_count; l++) {
            o_ptr_shifted[l] = result[l];
        }
        o_ptr_shifted += simd_element_count;
    }

    std::vector<scalar_t> result = std::vector<scalar_t>(size_remainder, 0.0);
    scalar_t* shifted_a_ptr = remainder_a_ptr;
    int32_t* p_ptr_temp = p_ptr;
    for (size_t j = 0; j < n_monomials; j++) {
        std::vector<scalar_t> temp = std::vector<scalar_t>(size_remainder, c_ptr[j]);
        for (size_t k = 0; k < polynomial_order; k++) {
            scalar_t* shifted_a_ptr_simd = shifted_a_ptr + p_ptr_temp[k]*size_remainder;
            for (size_t l = 0; l < size_remainder; l++) {
                temp[l] *= shifted_a_ptr_simd[l];
            }
        }
        for (size_t l = 0; l < size_remainder; l++) {
            result[l] += temp[l];
        }
        p_ptr_temp += polynomial_order;
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
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 2> p
) {

    if (tensor_a.shape[0] != output.shape[0]) {
        throw std::runtime_error(
            "A and O tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(tensor_a.shape[0]) + " and " +
            std::to_string(output.shape[0])
        );
    }

    if (tensor_c.shape[0] != p.shape[0]) {
        throw std::runtime_error(
            "C and P tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(tensor_c.shape[0]) + " and " +
            std::to_string(p.shape[0])
        );
    }

    size_t polynomial_order = p.shape[1];

    if (polynomial_order <= 10) {
        const uint8_t polynomial_order_u8 = static_cast<uint8_t>(polynomial_order);
        switch (polynomial_order_u8) {
            case 0:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 0>(output, tensor_a, tensor_c, p);
                return;
            case 1:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 1>(output, tensor_a, tensor_c, p);
                return;
            case 2:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 2>(output, tensor_a, tensor_c, p);
                return;
            case 3:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 3>(output, tensor_a, tensor_c, p);
                return;
            case 4:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 4>(output, tensor_a, tensor_c, p);
                return;
            case 5:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 5>(output, tensor_a, tensor_c, p);
                return;
            case 6:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 6>(output, tensor_a, tensor_c, p);
                return;
            case 7:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 7>(output, tensor_a, tensor_c, p);
                return;
            case 8:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 8>(output, tensor_a, tensor_c, p);
                return;
            case 9:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 9>(output, tensor_a, tensor_c, p);
                return;
            case 10:
                _homogeneous_polynomial_evaluation_templated_polynomial_order<scalar_t, 10>(output, tensor_a, tensor_c, p);
                return;
            default:
                break;
        }
    }

    throw std::runtime_error("Only up to polynomial order 10 is supported at the moment. Please contact the developers for more");
}
