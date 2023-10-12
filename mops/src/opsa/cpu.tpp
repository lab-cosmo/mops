#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>

#include "mops/opsa.hpp"

template<typename scalar_t>
void mops::outer_product_scatter_add(
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
