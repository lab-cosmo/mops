#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::sparse_accumulation_of_products(
    Tensor<scalar_t, 2> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 2> tensor_b,
    Tensor<int32_t, 1> indexes
) {
    if (tensor_a.shape[0] != tensor_b.shape[0]) {
        throw std::runtime_error(
            "A and B tensors must have the same number of elements along the "
            "first dimension, got " + std::to_string(tensor_a.shape[0]) + " and " +
            std::to_string(tensor_b.shape[0])
        );
    }

    if (tensor_a.shape[0] != indexes.shape[0]) {
        throw std::runtime_error(
            "indexes must contain the same number of elements as the first "
            "dimension of A and B , got " + std::to_string(indexes.shape[0]) +
            " and " + std::to_string(tensor_a.shape[0])
        );
    }

    if (tensor_a.shape[1] * tensor_b.shape[1] != output.shape[1]) {
        throw std::runtime_error(
            "output tensor must have space for " + std::to_string(tensor_a.shape[1] * tensor_b.shape[1]) +
            " along the second dimension, got " + std::to_string(output.shape[1])
        );
    }

    if (!std::is_sorted(indexes.data, indexes.data + indexes.shape[0])) {
        throw std::runtime_error("`indexes` values should be sorted");
    }

    for (size_t i=0; i<tensor_a.shape[0]; i++) {
        auto i_output = indexes.data[i];
        assert(i_output < output.shape[0]);
        for (size_t a_j=0; a_j<tensor_a.shape[1]; a_j++) {
            for (size_t b_j=0; b_j<tensor_b.shape[1]; b_j++) {
                auto output_index = tensor_b.shape[1] * (tensor_a.shape[1] * i_output + a_j) + b_j;
                output.data[output_index] += tensor_a.data[tensor_a.shape[1] * i + a_j]
                                           * tensor_b.data[tensor_b.shape[1] * i + b_j];
            }
        }
    }

    template<typename scalar_t>
    void _sparse_accumulation_active_dim_last_contiguous_forward(
        torch::Tensor output,
        torch::Tensor X1,
        torch::Tensor X2,
        torch::Tensor idx_output,
        int output_size,
        torch::Tensor idx_1,
        torch::Tensor idx_2,
        torch::Tensor multipliers) {
        
        scalar_t* X1_ptr = X1.data_ptr<scalar_t>();
        scalar_t* X2_ptr = X2.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        scalar_t* multipliers_ptr = multipliers.data_ptr<scalar_t>();
        long* idx_1_ptr = idx_1.data_ptr<long>();
        long* idx_2_ptr = idx_2.data_ptr<long>();
        long* idx_output_ptr = idx_output.data_ptr<long>();
        
        long active_size = idx_output.sizes()[0];
        long first_size = X1.sizes()[0];
        long second_size = X1.sizes()[1];
        
        long output_active_dim = output_size;
        long X1_active_dim = X1.sizes()[2];
        long X2_active_dim = X2.sizes()[2];
        
        #pragma omp parallel for
        for (int index_first = 0; index_first < first_size; ++index_first){
            // #pragma omp parallel for  // This makes little difference
            for (int index_second = 0; index_second < second_size; ++index_second) {
                long shift_number = index_first * second_size + index_second;
                long shift_output = shift_number * output_active_dim;
                long shift_X1 = shift_number * X1_active_dim;
                long shift_X2 = shift_number * X2_active_dim;
                for (int index = 0; index < active_size; ++index) { 
                    output_ptr[shift_output + idx_output_ptr[index]] += multipliers_ptr[index] * X1_ptr[shift_X1 + idx_1_ptr[index]] * X2_ptr[shift_X2 + idx_2_ptr[index]];                                             
                } 
            }
        }
            
    }
}



// if (tensor_a.requires_grad()) {
//         grad_a = torch::zeros_like(tensor_a);
//         scalar_t* grad_a_ptr = grad_a.data_ptr<scalar_t>();

//         #pragma omp parallel for
//         for (long idx_out = 0; idx_out < out_dim; idx_out++) {
//             long idx_in = first_occurrences_ptr[idx_out];
//             if (idx_in < 0) continue;
//             while (scatter_indices_ptr[idx_in] == idx_out) {
//                 for (long idx_a = 0; idx_a < size_a; idx_a++) {
//                     for (long idx_b = 0; idx_b < size_b; idx_b++) {
//                         grad_a_ptr[size_a*idx_in+idx_a] += grad_output_ptr[size_a*size_b*idx_out+size_b*idx_a+idx_b] * tensor_b_ptr[size_b*idx_in+idx_b];
//                     }
//                 }
//                 idx_in++;
//                 if (idx_in == size_scatter) break;
//             }
//         }
//     }

//     if (tensor_b.requires_grad()) {
//         grad_b = torch::zeros_like(tensor_b);
//         scalar_t* grad_b_ptr = grad_b.data_ptr<scalar_t>();

//         #pragma omp parallel for
//         for (long idx_out = 0; idx_out < out_dim; idx_out++) {
//             long idx_in = first_occurrences_ptr[idx_out];
//             if (idx_in < 0) continue;
//             while (scatter_indices_ptr[idx_in] == idx_out) {
//                 for (long idx_a = 0; idx_a < size_a; idx_a++) {
//                     for (long idx_b = 0; idx_b < size_b; idx_b++) {
//                         grad_b_ptr[size_b*idx_in+idx_b] += grad_output_ptr[size_a*size_b*idx_out+size_b*idx_a+idx_b] * tensor_a_ptr[size_a*idx_in+idx_a];
//                     }
//                 }
//                 idx_in++;
//                 if (idx_in == size_scatter) break;
//             }
//         }
//     }
