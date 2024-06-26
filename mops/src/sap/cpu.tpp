#include <algorithm>
#include <array>

#include "mops/sap.hpp"

#include "internal/checks/sap.hpp"
#include "internal/utils.hpp"


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
    check_sap(output, A, B, C, indices_A, indices_B, indices_output, "cpu_sparse_accumulation_of_products");

    scalar_t* c_ptr = C.data;
    int32_t* indices_A_ptr = indices_A.data;
    int32_t* indices_B_ptr = indices_B.data;
    int32_t* indices_output_ptr = indices_output.data;

    size_t size_first_dimension = A.shape[0];
    size_t size_second_dimension_a = A.shape[1];
    size_t size_second_dimension_b = B.shape[1];

    size_t size_second_dimension_o = output.shape[1];
    size_t c_size = C.shape[0];

    // The computation is parallel across the first dimension of A, B, and output. However,
    // the current layout, where the first dimension is the outermost, is not efficient for
    // SIMD operations. Therefore, we move (interleave) some of the first dimension into
    // an inner dimension. The chunk that is moved is the size of a 256-bit SIMD register,
    // as given by simd_element_count. The remainder is handled separately.

    constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();
    size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
    size_t size_remainder = size_first_dimension%simd_element_count;

    std::vector<scalar_t> interleft_o(size_first_dimension_interleft*size_second_dimension_o*simd_element_count);
    std::vector<scalar_t> remainder_o(size_remainder*size_second_dimension_o);
    scalar_t* interleft_o_ptr = interleft_o.data();
    scalar_t* remainder_o_ptr = remainder_o.data();

    std::vector<scalar_t> interleft_a(size_first_dimension_interleft*size_second_dimension_a*simd_element_count);
    std::vector<scalar_t> remainder_a(size_remainder*size_second_dimension_a);
    scalar_t* interleft_a_ptr = interleft_a.data();
    scalar_t* remainder_a_ptr = remainder_a.data();
    interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);

    std::vector<scalar_t> interleft_b(size_first_dimension_interleft*size_second_dimension_b*simd_element_count);
    std::vector<scalar_t> remainder_b(size_remainder*size_second_dimension_b);
    scalar_t* interleft_b_ptr = interleft_b.data();
    scalar_t* remainder_b_ptr = remainder_b.data();
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
    check_sap_vjp(grad_A, grad_B, grad_output, A, B, C, indices_A, indices_B, indices_output, "cpu_sparse_accumulation_of_products_vjp");

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

        std::vector<scalar_t> interleft_grad_o(size_first_dimension_interleft*size_second_dimension_o*simd_element_count);
        std::vector<scalar_t> remainder_grad_o(size_remainder*size_second_dimension_o);
        scalar_t* interleft_grad_o_ptr = interleft_grad_o.data();
        scalar_t* remainder_grad_o_ptr = remainder_grad_o.data();
        interleave_tensor<scalar_t, simd_element_count>(grad_output, interleft_grad_o_ptr, remainder_grad_o_ptr);

        scalar_t* interleft_a_ptr = nullptr;
        scalar_t* interleft_b_ptr = nullptr;
        scalar_t* remainder_a_ptr = nullptr;
        scalar_t* remainder_b_ptr = nullptr;
        scalar_t* interleft_grad_a_ptr = nullptr;
        scalar_t* interleft_grad_b_ptr = nullptr;
        scalar_t* remainder_grad_a_ptr = nullptr;
        scalar_t* remainder_grad_b_ptr = nullptr;

        std::vector<scalar_t> interleft_b;
        std::vector<scalar_t> remainder_b;
        std::vector<scalar_t> interleft_grad_a;
        std::vector<scalar_t> remainder_grad_a;
        if (calculate_grad_A) {
            interleft_b.resize(size_first_dimension_interleft*size_second_dimension_b*simd_element_count, static_cast<scalar_t>(0.0));
            remainder_b.resize(size_remainder*size_second_dimension_b, static_cast<scalar_t>(0.0));
            interleft_b_ptr = interleft_b.data();
            remainder_b_ptr = remainder_b.data();
            interleave_tensor<scalar_t, simd_element_count>(B, interleft_b_ptr, remainder_b_ptr);

            interleft_grad_a.resize(size_first_dimension_interleft*size_second_dimension_a*simd_element_count, static_cast<scalar_t>(0.0));
            remainder_grad_a.resize(size_remainder*size_second_dimension_a, static_cast<scalar_t>(0.0));
            interleft_grad_a_ptr = interleft_grad_a.data();
            remainder_grad_a_ptr = remainder_grad_a.data();
            std::fill(interleft_grad_a_ptr, interleft_grad_a_ptr+size_first_dimension_interleft*size_second_dimension_a*simd_element_count, static_cast<scalar_t>(0.0));
            std::fill(remainder_grad_a_ptr, remainder_grad_a_ptr+size_remainder*size_second_dimension_a, static_cast<scalar_t>(0.0));
        }

        std::vector<scalar_t> interleft_a;
        std::vector<scalar_t> remainder_a;
        std::vector<scalar_t> interleft_grad_b;
        std::vector<scalar_t> remainder_grad_b;
        if (calculate_grad_B) {
            interleft_a.resize(size_first_dimension_interleft*size_second_dimension_a*simd_element_count, static_cast<scalar_t>(0.0));
            remainder_a.resize(size_remainder*size_second_dimension_a, static_cast<scalar_t>(0.0));
            interleft_a_ptr = interleft_a.data();
            remainder_a_ptr = remainder_a.data();
            interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);

            interleft_grad_b.resize(size_first_dimension_interleft*size_second_dimension_b*simd_element_count, static_cast<scalar_t>(0.0));
            remainder_grad_b.resize(size_remainder*size_second_dimension_b, static_cast<scalar_t>(0.0));
            interleft_grad_b_ptr = interleft_grad_b.data();
            remainder_grad_b_ptr = remainder_grad_b.data();
            std::fill(interleft_grad_b_ptr, interleft_grad_b_ptr+size_first_dimension_interleft*size_second_dimension_b*simd_element_count, static_cast<scalar_t>(0.0));
            std::fill(remainder_grad_b_ptr, remainder_grad_b_ptr+size_remainder*size_second_dimension_b, static_cast<scalar_t>(0.0));
        }

        #pragma omp parallel for
        for (size_t i = 0; i < size_first_dimension_interleft; i++){
            scalar_t* grad_output_i = interleft_grad_o_ptr + i * size_second_dimension_o * simd_element_count;
            scalar_t* grad_a_i = nullptr;
            scalar_t* b_i = nullptr;
            if (calculate_grad_A) {
                grad_a_i = interleft_grad_a_ptr + i * size_second_dimension_a * simd_element_count;
                b_i = interleft_b_ptr + i * size_second_dimension_b * simd_element_count;
            }
            scalar_t* grad_b_i = nullptr;
            scalar_t* a_i = nullptr;
            if (calculate_grad_B) {
                grad_b_i = interleft_grad_b_ptr + i * size_second_dimension_b * simd_element_count;
                a_i = interleft_a_ptr + i * size_second_dimension_a * simd_element_count;
            }
            for (size_t j = 0; j < c_size; j++) {
                scalar_t* grad_output_i_j = grad_output_i + p_o_ptr[j] * simd_element_count;
                std::array<scalar_t, simd_element_count> common_factor;
                scalar_t c_ptr_j = c_ptr[j];
                for (size_t l = 0; l < simd_element_count; l++) {
                    common_factor[l] = c_ptr_j * grad_output_i_j[l];
                }
                if (calculate_grad_A) {
                    scalar_t* grad_a_i_j = grad_a_i + p_a_ptr[j] * simd_element_count;
                    scalar_t* b_i_j = b_i + p_b_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_a_i_j[l] += common_factor[l] * b_i_j[l];
                    }
                }
                if (calculate_grad_B) {
                    scalar_t* grad_b_i_j = grad_b_i + p_b_ptr[j] * simd_element_count;
                    scalar_t* a_i_j = a_i + p_a_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_b_i_j[l] += common_factor[l] * a_i_j[l];
                    }
                }
            }
        }

        // Handle the remainder, i.e., the elements that do not fit inside a multiple of simd_element_count
        for (size_t j = 0; j < c_size; j++) {
            scalar_t* grad_output_j = remainder_grad_o_ptr + p_o_ptr[j] * size_remainder;
            scalar_t* grad_a_j = nullptr;
            if (calculate_grad_A) {
                grad_a_j = remainder_grad_a_ptr + p_a_ptr[j] * size_remainder;
            }
            scalar_t* grad_b_j = nullptr;
            if (calculate_grad_B) {
                grad_b_j = remainder_grad_b_ptr + p_b_ptr[j] * size_remainder;
            }
            scalar_t* a_j = remainder_a_ptr + p_a_ptr[j] * size_remainder;
            scalar_t* b_j = remainder_b_ptr + p_b_ptr[j] * size_remainder;
            scalar_t C_j = c_ptr[j];
            for (size_t k = 0; k < size_remainder; k++) {
                scalar_t grad_output_j_k = grad_output_j[k];
                scalar_t common_factor = grad_output_j_k * C_j;
                if (calculate_grad_A) {
                    grad_a_j[k] += common_factor * b_j[k];
                }
                if (calculate_grad_B) {
                    grad_b_j[k] += common_factor * a_j[k];
                }
            }
        }

        if (calculate_grad_A) {
            un_interleave_tensor<scalar_t, simd_element_count>(grad_A, interleft_grad_a_ptr, remainder_grad_a_ptr);
        }
        if (calculate_grad_B) {
            un_interleave_tensor<scalar_t, simd_element_count>(grad_B, interleft_grad_b_ptr, remainder_grad_b_ptr);
        }
    }

}

template<typename scalar_t>
void mops::sparse_accumulation_of_products_vjp_vjp(
    Tensor<scalar_t, 2> grad_grad_output,
    Tensor<scalar_t, 2> grad_A_2,
    Tensor<scalar_t, 2> grad_B_2,
    Tensor<scalar_t, 2> grad_grad_A,
    Tensor<scalar_t, 2> grad_grad_B,
    Tensor<scalar_t, 2> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
) {
    // This function computes the vjp of sparse_accumulation_of_products_vjp.
    // Therefore, its aim is to compute the gradient of the outputs of
    // sparse_accumulation_of_products_vjp, that is grad_A and grad_B, with respect to its
    // differentiable inputs, that is grad_output, A and B. Hence, the inputs of this
    // function are the same as those to sparse_accumulation_of_products_vjp
    // (A, B, C, indices_A, indices_B, indices_output, grad_output) plus the gradient of
    // the scalar objective with respect to its outputs grad_A and grad_B, that is,
    // grad_grad_A and grad_grad_B. Our objective is to fill the derivatives of the scalar
    // objective with respect to the differentiable inputs of sparse_accumulation_of_products_vjp,
    // which are represented by grad_A_2 and grad_B_2 (derivative of A and B, not to be confused
    // with grad_A and grad_B, which is the outputs of the function sparse_accumulation_of_products_vjp),
    // as well as grad_grad_output(derivative of the scalar objective with respect to grad_output).

    // First, we interleave the inputs to facilitate SIMD operations (similar to
    // sparse_accumulation_of_products_vjp). Then, we compute the derivatives.
    // The many branches of the code take into account whether grad_grad_A and grad_grad_B are available
    // (they might not be if grad_A/grad_B was not computed in sparse_accumulation_of_products_vjp),
    // and whether the user wants to compute grad_A_2. grad_B_2 and/or grad_grad_output.
    // Finally, we un-interleave the results to return the outputs in the correct format.

    check_sap_vjp_vjp(grad_grad_output, grad_A_2, grad_B_2, grad_grad_A, grad_grad_B, grad_output, A, B, C, indices_A, indices_B, indices_output, "cpu_sparse_accumulation_of_products_vjp_vjp");

    bool grad_grad_A_is_available = (grad_grad_A.data != nullptr);
    bool grad_grad_B_is_available = (grad_grad_B.data != nullptr);

    bool calculate_grad_grad_output = (grad_grad_output.data != nullptr);
    bool calculate_grad_A_2 = (grad_A_2.data != nullptr);
    bool calculate_grad_B_2 = (grad_B_2.data != nullptr);

    size_t size_first_dimension = A.shape[0];
    size_t size_second_dimension_a = A.shape[1];
    size_t size_second_dimension_b = B.shape[1];
    size_t size_second_dimension_o = grad_output.shape[1];
    size_t c_size = C.shape[0];

    scalar_t* c_ptr = C.data;
    int32_t* p_a_ptr = indices_A.data;
    int32_t* p_b_ptr = indices_B.data;
    int32_t* p_o_ptr = indices_output.data;

    constexpr size_t simd_element_count = get_simd_element_count<scalar_t>();
    size_t size_first_dimension_interleft = size_first_dimension/simd_element_count;
    size_t size_remainder = size_first_dimension%simd_element_count;

    std::vector<scalar_t> interleft_grad_grad_a;
    std::vector<scalar_t> remainder_grad_grad_a;
    scalar_t* interleft_grad_grad_a_ptr = nullptr;
    scalar_t* remainder_grad_grad_a_ptr = nullptr;
    if (grad_grad_A_is_available) {
        interleft_grad_grad_a.resize(size_first_dimension_interleft*size_second_dimension_a*simd_element_count);
        remainder_grad_grad_a.resize(size_remainder*size_second_dimension_a);
        interleft_grad_grad_a_ptr = interleft_grad_grad_a.data();
        remainder_grad_grad_a_ptr = remainder_grad_grad_a.data();
        interleave_tensor<scalar_t, simd_element_count>(grad_grad_A, interleft_grad_grad_a_ptr, remainder_grad_grad_a_ptr);
    }

    std::vector<scalar_t> interleft_b;
    std::vector<scalar_t> remainder_b;
    scalar_t* interleft_b_ptr = nullptr;
    scalar_t* remainder_b_ptr = nullptr;
    if (grad_grad_A_is_available && calculate_grad_grad_output) {
        interleft_b.resize(size_first_dimension_interleft*size_second_dimension_b*simd_element_count);
        remainder_b.resize(size_remainder*size_second_dimension_b);
        interleft_b_ptr = interleft_b.data();
        remainder_b_ptr = remainder_b.data();
        interleave_tensor<scalar_t, simd_element_count>(B, interleft_b_ptr, remainder_b_ptr);
    }

    std::vector<scalar_t> interleft_a;
    std::vector<scalar_t> remainder_a;
    scalar_t* interleft_a_ptr = nullptr;
    scalar_t* remainder_a_ptr = nullptr;
    if (grad_grad_B_is_available && calculate_grad_grad_output) {
        interleft_a.resize(size_first_dimension_interleft*size_second_dimension_a*simd_element_count);
        remainder_a.resize(size_remainder*size_second_dimension_a);
        interleft_a_ptr = interleft_a.data();
        remainder_a_ptr = remainder_a.data();
        interleave_tensor<scalar_t, simd_element_count>(A, interleft_a_ptr, remainder_a_ptr);
    }

    std::vector<scalar_t> interleft_grad_grad_b;
    std::vector<scalar_t> remainder_grad_grad_b;
    scalar_t* interleft_grad_grad_b_ptr = nullptr;
    scalar_t* remainder_grad_grad_b_ptr = nullptr;
    if (grad_grad_B_is_available) {
        interleft_grad_grad_b.resize(size_first_dimension_interleft*size_second_dimension_b*simd_element_count);
        remainder_grad_grad_b.resize(size_remainder*size_second_dimension_b);
        interleft_grad_grad_b_ptr = interleft_grad_grad_b.data();
        remainder_grad_grad_b_ptr = remainder_grad_grad_b.data();
        interleave_tensor<scalar_t, simd_element_count>(grad_grad_B, interleft_grad_grad_b_ptr, remainder_grad_grad_b_ptr);
    }

    std::vector<scalar_t> interleft_grad_output;
    std::vector<scalar_t> remainder_grad_output;
    scalar_t* interleft_grad_output_ptr = nullptr;
    scalar_t* remainder_grad_output_ptr = nullptr;
    // this one is almost always needed
    interleft_grad_output.resize(size_first_dimension_interleft*size_second_dimension_o*simd_element_count);
    remainder_grad_output.resize(size_remainder*size_second_dimension_o);
    interleft_grad_output_ptr = interleft_grad_output.data();
    remainder_grad_output_ptr = remainder_grad_output.data();
    interleave_tensor<scalar_t, simd_element_count>(grad_output, interleft_grad_output_ptr, remainder_grad_output_ptr);

    std::vector<scalar_t> interleft_grad_grad_output;
    std::vector<scalar_t> remainder_grad_grad_output;
    scalar_t* interleft_grad_grad_output_ptr = nullptr;
    scalar_t* remainder_grad_grad_output_ptr = nullptr;
    if (calculate_grad_grad_output) {
        interleft_grad_grad_output.resize(size_first_dimension_interleft*size_second_dimension_o*simd_element_count, static_cast<scalar_t>(0.0));
        remainder_grad_grad_output.resize(size_remainder*size_second_dimension_o, static_cast<scalar_t>(0.0));
        interleft_grad_grad_output_ptr = interleft_grad_grad_output.data();
        remainder_grad_grad_output_ptr = remainder_grad_grad_output.data();
    }

    std::vector<scalar_t> interleft_grad_a_2;
    std::vector<scalar_t> remainder_grad_a_2;
    scalar_t* interleft_grad_a_2_ptr = nullptr;
    scalar_t* remainder_grad_a_2_ptr = nullptr;
    if (calculate_grad_A_2) {
        interleft_grad_a_2.resize(size_first_dimension_interleft*size_second_dimension_a*simd_element_count, static_cast<scalar_t>(0.0));
        remainder_grad_a_2.resize(size_remainder*size_second_dimension_a, static_cast<scalar_t>(0.0));
        interleft_grad_a_2_ptr = interleft_grad_a_2.data();
        remainder_grad_a_2_ptr = remainder_grad_a_2.data();
    }

    std::vector<scalar_t> interleft_grad_b_2;
    std::vector<scalar_t> remainder_grad_b_2;
    scalar_t* interleft_grad_b_2_ptr = nullptr;
    scalar_t* remainder_grad_b_2_ptr = nullptr;
    if (calculate_grad_B_2) {
        interleft_grad_b_2.resize(size_first_dimension_interleft*size_second_dimension_b*simd_element_count, static_cast<scalar_t>(0.0));
        remainder_grad_b_2.resize(size_remainder*size_second_dimension_b, static_cast<scalar_t>(0.0));
        interleft_grad_b_2_ptr = interleft_grad_b_2.data();
        remainder_grad_b_2_ptr = remainder_grad_b_2.data();
    }

    #pragma omp parallel for
    for (size_t i = 0; i < size_first_dimension_interleft; i++){
        scalar_t* grad_grad_output_i = nullptr;
        if (calculate_grad_grad_output) {
            grad_grad_output_i = interleft_grad_grad_output_ptr + i * size_second_dimension_o * simd_element_count;
        }
        scalar_t* grad_a_2_i = nullptr;
        if (calculate_grad_A_2) {
            grad_a_2_i = interleft_grad_a_2_ptr + i * size_second_dimension_a * simd_element_count;
        }
        scalar_t* grad_b_2_i = nullptr;
        if (calculate_grad_B_2) {
            grad_b_2_i = interleft_grad_b_2_ptr + i * size_second_dimension_b * simd_element_count;
        }
        scalar_t* grad_grad_a_i = nullptr;
        if (grad_grad_A_is_available) {
            grad_grad_a_i = interleft_grad_grad_a_ptr + i * size_second_dimension_a * simd_element_count;
        }
        scalar_t* grad_grad_b_i = nullptr;
        if (grad_grad_B_is_available) {
            grad_grad_b_i = interleft_grad_grad_b_ptr + i * size_second_dimension_b * simd_element_count;
        }
        scalar_t* grad_output_i = interleft_grad_output_ptr + i * size_second_dimension_o * simd_element_count;
        scalar_t* a_i = nullptr;
        if (grad_grad_B_is_available && calculate_grad_grad_output) {
            a_i = interleft_a_ptr + i * size_second_dimension_a * simd_element_count;
        }
        scalar_t* b_i = nullptr;
        if (grad_grad_A_is_available && calculate_grad_grad_output) {
            b_i = interleft_b_ptr + i * size_second_dimension_b * simd_element_count;
        }
        for (size_t j = 0; j < c_size; j++) {
            scalar_t C_j = c_ptr[j];
            if (grad_grad_A_is_available) {
                if (calculate_grad_grad_output) {
                    scalar_t* grad_grad_output_i_j = grad_grad_output_i + p_o_ptr[j] * simd_element_count;
                    scalar_t* grad_grad_a_i_j = grad_grad_a_i + p_a_ptr[j] * simd_element_count;
                    scalar_t* b_i_j = b_i + p_b_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_grad_output_i_j[l] += C_j * grad_grad_a_i_j[l] * b_i_j[l];
                    }
                }
                if (calculate_grad_B_2) {
                    scalar_t* grad_b_2_i_j = grad_b_2_i + p_b_ptr[j] * simd_element_count;
                    scalar_t* grad_grad_a_i_j = grad_grad_a_i + p_a_ptr[j] * simd_element_count;
                    scalar_t* grad_output_i_j = grad_output_i + p_o_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_b_2_i_j[l] += C_j * grad_grad_a_i_j[l] * grad_output_i_j[l];
                    }
                }
            }
            if (grad_grad_B_is_available) {
                if (calculate_grad_grad_output) {
                    scalar_t* grad_grad_output_i_j = grad_grad_output_i + p_o_ptr[j] * simd_element_count;
                    scalar_t* grad_grad_b_i_j = grad_grad_b_i + p_b_ptr[j] * simd_element_count;
                    scalar_t* a_i_j = a_i + p_a_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_grad_output_i_j[l] += C_j * grad_grad_b_i_j[l] * a_i_j[l];
                    }
                }
                if (calculate_grad_A_2) {
                    scalar_t* grad_a_2_i_j = grad_a_2_i + p_a_ptr[j] * simd_element_count;
                    scalar_t* grad_grad_b_i_j = grad_grad_b_i + p_b_ptr[j] * simd_element_count;
                    scalar_t* grad_output_i_j = grad_output_i + p_o_ptr[j] * simd_element_count;
                    for (size_t l = 0; l < simd_element_count; l++) {
                        grad_a_2_i_j[l] += C_j * grad_grad_b_i_j[l] * grad_output_i_j[l];
                    }
                }
            }
        }
    }

    // Handle the remainder, i.e., the elements that do not fit inside a multiple of simd_element_count
    for (size_t j = 0; j < c_size; j++) {
        scalar_t C_j = c_ptr[j];
        scalar_t* grad_grad_output_j = nullptr;
        if (calculate_grad_grad_output) {
            grad_grad_output_j = remainder_grad_grad_output_ptr + p_o_ptr[j] * size_remainder;
        }
        scalar_t* grad_a_2_j = nullptr;
        if (calculate_grad_A_2) {
            grad_a_2_j = remainder_grad_a_2_ptr + p_a_ptr[j] * size_remainder;
        }
        scalar_t* grad_b_2_j = nullptr;
        if (calculate_grad_B_2) {
            grad_b_2_j = remainder_grad_b_2_ptr + p_b_ptr[j] * size_remainder;
        }
        scalar_t* grad_grad_a_j = nullptr;
        if (grad_grad_A_is_available) {
            grad_grad_a_j = remainder_grad_grad_a_ptr + p_a_ptr[j] * size_remainder;
        }
        scalar_t* grad_grad_b_j = nullptr;
        if (grad_grad_B_is_available) {
            grad_grad_b_j = remainder_grad_grad_b_ptr + p_b_ptr[j] * size_remainder;
        }
        scalar_t* grad_output_j = remainder_grad_output_ptr + p_o_ptr[j] * size_remainder;
        scalar_t* a_j = nullptr;
        if (grad_grad_B_is_available && calculate_grad_grad_output) {
            a_j = remainder_a_ptr + p_a_ptr[j] * size_remainder;
        }
        scalar_t* b_j = nullptr;
        if (grad_grad_A_is_available && calculate_grad_grad_output) {
            b_j = remainder_b_ptr + p_b_ptr[j] * size_remainder;
        }
        for (size_t k = 0; k < size_remainder; k++) {
            if (grad_grad_A_is_available) {
                if (calculate_grad_grad_output) {
                    grad_grad_output_j[k] += C_j * grad_grad_a_j[k] * b_j[k];
                }
                if (calculate_grad_B_2) {
                    grad_b_2_j[k] += C_j * grad_grad_a_j[k] * grad_output_j[k];
                }
            }
            if (grad_grad_B_is_available) {
                if (calculate_grad_grad_output) {
                    grad_grad_output_j[k] += C_j * grad_grad_b_j[k] * a_j[k];
                }
                if (calculate_grad_A_2) {
                    grad_a_2_j[k] += C_j * grad_grad_b_j[k] * grad_output_j[k];
                }
            }
        }
    }

    if (calculate_grad_grad_output) {
        un_interleave_tensor<scalar_t, simd_element_count>(grad_grad_output, interleft_grad_grad_output_ptr, remainder_grad_grad_output_ptr);
    }
    if (calculate_grad_A_2) {
        un_interleave_tensor<scalar_t, simd_element_count>(grad_A_2, interleft_grad_a_2_ptr, remainder_grad_a_2_ptr);
    }
    if (calculate_grad_B_2) {
        un_interleave_tensor<scalar_t, simd_element_count>(grad_B_2, interleft_grad_b_2_ptr, remainder_grad_b_2_ptr);
    }
}
