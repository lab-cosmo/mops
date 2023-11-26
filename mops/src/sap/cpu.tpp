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
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    check_sizes(A, "A", 0, B, "B", 0, "sap");
    check_sizes(A, "A", 0, output, "O", 0, "sap");
    check_sizes(C, "C", 0, indices_A, "indices_A", 0, "sap");
    check_sizes(C, "C", 0, indices_B, "indices_B", 0, "sap");
    check_sizes(C, "C", 0, indices_output, "indices_output", 0, "sap");
    check_index_tensor(indices_A, "indices_A", A.shape[1], "sap");
    check_index_tensor(indices_B, "indices_B", B.shape[1], "sap");
    check_index_tensor(indices_output, "indices_output", output.shape[1], "sap");

    // if (!std::is_sorted(indices_output.data, indices_output.data + indices_output.shape[0])) {
    //     throw std::runtime_error("indices_output values should be sorted");
    // }

    scalar_t* o_ptr = output.data;
    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;
    int32_t* indices_B_ptr = indices_B.data;
    int32_t* indices_output_ptr = indices_output.data;

    size_t size_first_dimension = A.shape[0];
    size_t size_second_dimension_a = A.shape[1];
    size_t size_second_dimension_b = B.shape[1];

    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = C.shape[0];

    std::fill(o_ptr, o_ptr+output.shape[0]*output.shape[1], static_cast<scalar_t>(0.0));

    constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();
    size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
    size_t size_remainder = size_first_dimension%simd_element_count;

    scalar_t* interleft_o_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_o*simd_element_count];
    scalar_t* remainder_o_ptr = new scalar_t[size_remainder*size_second_dimension_o];

    scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
    scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
    interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);

    scalar_t* interleft_b_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_b*simd_element_count];
    scalar_t* remainder_b_ptr = new scalar_t[size_remainder*size_second_dimension_b];
    interleave_tensor<scalar_t, simd_element_count>(B, interleft_b_ptr, remainder_b_ptr);

    std::fill(interleft_o_ptr, interleft_o_ptr+size_first_dimension_interleft*size_second_dimension_o*simd_element_count, static_cast<scalar_t>(0.0));
    std::fill(remainder_o_ptr, remainder_o_ptr+size_remainder*size_second_dimension_o, static_cast<scalar_t>(0.0));
    
    #pragma omp parallel for
    for (size_t i = 0; i < size_first_dimension_interleft; i++) {
        scalar_t* a_ptr_shifted_first_dim = interleft_a_ptr + i * size_second_dimension_a*simd_element_count;
        scalar_t* b_ptr_shifted_first_dim = interleft_b_ptr + i * size_second_dimension_b*simd_element_count;
        scalar_t* o_ptr_shifted_first_dim = interleft_o_ptr + i * size_second_dimension_o*simd_element_count;
        for (size_t j = 0; j < c_size; j++) {
            scalar_t* a_ptr_shifted_second_dim = a_ptr_shifted_first_dim + indices_A_ptr[j] * simd_element_count;
            scalar_t* b_ptr_shifted_second_dim = b_ptr_shifted_first_dim + indices_B_ptr[j] * simd_element_count;
            scalar_t* o_ptr_shifted_second_dim = o_ptr_shifted_first_dim + indices_output_ptr[j] * simd_element_count;
            scalar_t current_c = c_ptr[j];
            for (size_t k = 0; k < simd_element_count; k++) {
                o_ptr_shifted_second_dim[k] += current_c * a_ptr_shifted_second_dim[k] * b_ptr_shifted_second_dim[k];
            }                                             
        }
    }

    // Handle remainder
    for (size_t j = 0; j < c_size; j++) {
        scalar_t* a_ptr_shifted_second_dim = remainder_a_ptr + indices_A_ptr[j] * size_remainder;
        scalar_t* b_ptr_shifted_second_dim = remainder_b_ptr + indices_B_ptr[j] * size_remainder;
        scalar_t* o_ptr_shifted_second_dim = remainder_o_ptr + indices_output_ptr[j] * size_remainder;
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


template<typename scalar_t>
void mops::sparse_accumulation_of_products_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    // TODO: checks

    bool calculate_grad_A = grad_A.data != nullptr;
    bool calculate_grad_B = grad_B.data != nullptr;

    if (calculate_grad_A || calculate_grad_B) {

        scalar_t* c_ptr = C.data;
        int32_t* p_a_ptr = indices_A.data;
        int32_t* p_b_ptr = indices_B.data;
        int32_t* p_o_ptr = indices_output.data;

        size_t size_first_dimension = A.shape[0];
        size_t size_second_dimension_a = A.shape[1];
        size_t size_second_dimension_b = B.shape[1];
        size_t size_second_dimension_o = grad_output.shape[1];
        size_t c_size = C.shape[0];

        constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();
        size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
        size_t size_remainder = size_first_dimension%simd_element_count;

        // You don't always need all of these!! TODO

        scalar_t* interleft_grad_o_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_o*simd_element_count];
        scalar_t* remainder_grad_o_ptr = new scalar_t[size_remainder*size_second_dimension_o];
        interleave_tensor<scalar_t, simd_element_count>(grad_output, interleft_grad_o_ptr, remainder_grad_o_ptr);

        scalar_t* interleft_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
        scalar_t* remainder_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
        interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);

        scalar_t* interleft_b_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_b*simd_element_count];
        scalar_t* remainder_b_ptr = new scalar_t[size_remainder*size_second_dimension_b];
        interleave_tensor<scalar_t, simd_element_count>(B, interleft_b_ptr, remainder_b_ptr);

        scalar_t* interleft_grad_a_ptr = nullptr;
        scalar_t* interleft_grad_b_ptr = nullptr;
        scalar_t* remainder_grad_a_ptr = nullptr;
        scalar_t* remainder_grad_b_ptr = nullptr;
        if (calculate_grad_A) {
            interleft_grad_a_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_a*simd_element_count];
            remainder_grad_a_ptr = new scalar_t[size_remainder*size_second_dimension_a];
            std::fill(interleft_grad_a_ptr, interleft_grad_a_ptr+size_first_dimension_interleft*size_second_dimension_a*simd_element_count, static_cast<scalar_t>(0.0));
            std::fill(remainder_grad_a_ptr, remainder_grad_a_ptr+size_remainder*size_second_dimension_a, static_cast<scalar_t>(0.0));
        }
        if (calculate_grad_B) {
            interleft_grad_b_ptr = new scalar_t[size_first_dimension_interleft*size_second_dimension_b*simd_element_count];
            remainder_grad_b_ptr = new scalar_t[size_remainder*size_second_dimension_b];
            std::fill(interleft_grad_b_ptr, interleft_grad_b_ptr+size_first_dimension_interleft*size_second_dimension_b*simd_element_count, static_cast<scalar_t>(0.0));
            std::fill(remainder_grad_b_ptr, remainder_grad_b_ptr+size_remainder*size_second_dimension_b, static_cast<scalar_t>(0.0));
        }

        scalar_t* grad_output_row = interleft_grad_o_ptr; 
        scalar_t* grad_a_row = interleft_grad_a_ptr;
        scalar_t* grad_b_row = interleft_grad_b_ptr;
        scalar_t* a_row = interleft_a_ptr;
        scalar_t* b_row = interleft_b_ptr;
        for (size_t i = 0; i < size_first_dimension_interleft; i++){
            for (size_t j = 0; j < c_size; j++) {
                scalar_t* grad_output_row_j = grad_output_row + p_o_ptr[j] * simd_element_count;      
                std::array<scalar_t, simd_element_count> common_factor; 
                scalar_t c_ptr_j = c_ptr[j];
                for (size_t l = 0; l < simd_element_count; l++) common_factor[l] = c_ptr_j * grad_output_row_j[l];
                if (calculate_grad_A) {
                    scalar_t* grad_a_row_j = grad_a_row + p_a_ptr[j] * simd_element_count;
                    scalar_t* b_row_j = b_row + p_b_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) grad_a_row_j[l] += common_factor[l] * b_row_j[l];
                }
                if (calculate_grad_B) {
                    scalar_t* grad_b_row_j = grad_b_row + p_b_ptr[j] * simd_element_count;
                    scalar_t* a_row_j = a_row + p_a_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) grad_b_row_j[l] += common_factor[l] * a_row_j[l];
                }
            }
            grad_output_row += size_second_dimension_o * simd_element_count;
            grad_a_row += size_second_dimension_a * simd_element_count;
            grad_b_row += size_second_dimension_b * simd_element_count;
            a_row += size_second_dimension_a * simd_element_count;
            b_row += size_second_dimension_b * simd_element_count;
        }

        grad_output_row = remainder_grad_o_ptr; 
        grad_a_row = remainder_grad_a_ptr;
        grad_b_row = remainder_grad_b_ptr;
        a_row = remainder_a_ptr;
        b_row = remainder_b_ptr;
        for (size_t i = 0; i < size_remainder; i++){
            for (size_t j = 0; j < c_size; j++) {                
                scalar_t grad_output_j = grad_output_row[p_o_ptr[j]];
                scalar_t common_factor = grad_output_j * c_ptr[j];
                if (calculate_grad_A) grad_a_row[p_a_ptr[j]] += common_factor * b_row[p_b_ptr[j]];
                if (calculate_grad_B) grad_b_row[p_b_ptr[j]] += common_factor * a_row[p_a_ptr[j]];
            }
            grad_output_row += size_second_dimension_o;
            grad_a_row += size_second_dimension_a;
            grad_b_row += size_second_dimension_b;
            a_row += size_second_dimension_a;
            b_row += size_second_dimension_b;
        }

        if (calculate_grad_A) {
            un_interleave_tensor<scalar_t, simd_element_count>(grad_A, interleft_grad_a_ptr, remainder_grad_a_ptr);
            delete[] interleft_grad_a_ptr;
            delete[] remainder_grad_a_ptr;
        }
        if (calculate_grad_B) {
            un_interleave_tensor<scalar_t, simd_element_count>(grad_B, interleft_grad_b_ptr, remainder_grad_b_ptr);
            delete[] interleft_grad_b_ptr;
            delete[] remainder_grad_b_ptr;
        }
    }

}
