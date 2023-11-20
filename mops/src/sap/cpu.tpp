#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/sap.hpp"
#include "mops/checks.hpp"
#include "mops/utils.hpp"


template<typename scalar_t>
void mops::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_b,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 1> p_a,
    Tensor<int32_t, 1> p_b,
    Tensor<int32_t, 1> p_o
) {
    check_sizes(tensor_a, "A", 0, tensor_b, "B", 0, "sap");
    check_sizes(tensor_a, "A", 0, output, "O", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_a, "P_A", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_b, "P_B", 0, "sap");
    check_sizes(tensor_c, "C", 0, p_o, "P_O", 0, "sap");
    check_index_tensor(p_a, "P_A", tensor_a.shape[1], "sap");
    check_index_tensor(p_b, "P_B", tensor_b.shape[1], "sap");
    check_index_tensor(p_o, "P_O", output.shape[1], "sap");

    if (!std::is_sorted(p_o.data, p_o.data + p_o.shape[0])) {
        throw std::runtime_error("P_O values should be sorted");
    }

    scalar_t* c_ptr = tensor_c.data;
    int32_t* p_a_ptr = p_a.data;
    int32_t* p_b_ptr = p_b.data;
    int32_t* p_o_ptr = p_o.data;

    size_t size_first_dimension = tensor_a.shape[0];
    size_t size_second_dimension_a = tensor_a.shape[1];
    size_t size_second_dimension_b = tensor_b.shape[1];
    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = tensor_c.shape[0];

    constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();

    size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
    size_t size_remainder = size_first_dimension%simd_element_count;

    scalar_t* interleft_o_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_o*simd_element_count];
    scalar_t* remainder_o_ptr = new scalar_t[size_remainder*size_second_dimension_o];

    scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
    scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
    interleave_tensor<scalar_t, simd_element_count>(tensor_a, interleft_a_ptr, remainder_a_ptr);

    scalar_t* interleft_b_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_b*simd_element_count];
    scalar_t* remainder_b_ptr = new scalar_t[size_remainder*size_second_dimension_b];
    interleave_tensor<scalar_t, simd_element_count>(tensor_b, interleft_b_ptr, remainder_b_ptr);

    std::fill(interleft_o_ptr, interleft_o_ptr+size_first_dimension_interleft*size_second_dimension_o*simd_element_count, static_cast<scalar_t>(0.0));
    std::fill(remainder_o_ptr, remainder_o_ptr+size_remainder*size_second_dimension_o, static_cast<scalar_t>(0.0));
    std::vector<int32_t> first_occurrences = find_first_occurrences(p_o_ptr, c_size, size_second_dimension_o);
    
    #pragma omp parallel for
    for (size_t i = 0; i < size_first_dimension_interleft; i++) {
        scalar_t* a_ptr_shifted_first_dim = interleft_a_ptr + i * size_second_dimension_a*simd_element_count;
        scalar_t* b_ptr_shifted_first_dim = interleft_b_ptr + i * size_second_dimension_b*simd_element_count;
        scalar_t* o_ptr_shifted_first_dim = interleft_o_ptr + i * size_second_dimension_o*simd_element_count;
        scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim;
        for (size_t ji = 0; ji < size_second_dimension_o; ji++) {
            for (int32_t j = first_occurrences[ji]; j < first_occurrences[ji+1]; j++) {
                scalar_t* a_ptr_shifted_second_dim = a_ptr_shifted_first_dim + p_a_ptr[j] * simd_element_count;
                scalar_t* b_ptr_shifted_second_dim = b_ptr_shifted_first_dim + p_b_ptr[j] * simd_element_count;
                scalar_t current_c = c_ptr[j];
                for (size_t k = 0; k < simd_element_count; k++) {
                    o_ptr_shifted_second_dim[k] += current_c * a_ptr_shifted_second_dim[k] * b_ptr_shifted_second_dim[k];
                }                                             
            }
            o_ptr_shifted_second_dim += simd_element_count;
        }
    }

    // Handle remainder
    for (size_t j = 0; j < c_size; j++) {
        scalar_t* a_ptr_shifted_second_dim = remainder_a_ptr + p_a_ptr[j] * size_remainder;
        scalar_t* b_ptr_shifted_second_dim = remainder_b_ptr + p_b_ptr[j] * size_remainder;
        scalar_t* o_ptr_shifted_second_dim = remainder_o_ptr + p_o_ptr[j] * size_remainder;
        scalar_t current_c = c_ptr[j];
        for (size_t k = 0; k < size_remainder; k++) {
            o_ptr_shifted_second_dim[k] += current_c * a_ptr_shifted_second_dim[k] * b_ptr_shifted_second_dim[k];
        }                                             
    }

    un_interleave_tensor<scalar_t, simd_element_count>(output, interleft_o_ptr, remainder_o_ptr);

    delete[] interleft_o_ptr;
    delete[] interleft_a_ptr;
    delete[] interleft_b_ptr;
    delete[] remainder_o_ptr;
    delete[] remainder_a_ptr;
    delete[] remainder_b_ptr;
}
