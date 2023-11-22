#ifndef MOPS_SASAX_HPP
#define MOPS_SASAX_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
    /// TODO
    template<typename scalar_t>
    void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights(
        Tensor<scalar_t, 3> output,
        Tensor<scalar_t, 2> tensor_a,
        Tensor<scalar_t, 2> tensor_r,
        Tensor<scalar_t, 3> tensor_x,
        Tensor<scalar_t, 1> tensor_c,
        Tensor<int, 1> tensor_i,
        Tensor<int, 1> tensor_j,
        Tensor<int, 1> tensor_m_1,
        Tensor<int, 1> tensor_m_2,
        Tensor<int, 1> tensor_m_3
    );

    // these templates will be precompiled and provided in the mops library
    extern template void sparse_accumulation_scatter_add_with_weights(
        Tensor<float, 3> output,
        Tensor<float, 2> tensor_a,
        Tensor<float, 2> tensor_r,
        Tensor<float, 3> tensor_x,
        Tensor<float, 1> tensor_c,
        Tensor<int, 1> tensor_i,
        Tensor<int, 1> tensor_j,
        Tensor<int, 1> tensor_m_1,
        Tensor<int, 1> tensor_m_2,
        Tensor<int, 1> tensor_m_3
    );

    extern template void sparse_accumulation_scatter_add_with_weights(
        Tensor<double, 3> output,
        Tensor<double, 2> tensor_a,
        Tensor<double, 2> tensor_r,
        Tensor<double, 3> tensor_x,
        Tensor<double, 1> tensor_c,
        Tensor<int, 1> tensor_i,
        Tensor<int, 1> tensor_j,
        Tensor<int, 1> tensor_m_1,
        Tensor<int, 1> tensor_m_2,
        Tensor<int, 1> tensor_m_3
    );

    namespace cuda {
        /// CUDA version of mops::sparse_accumulation_scatter_add_with
        template<typename scalar_t>
        void MOPS_EXPORT sparse_accumulation_scatter_add_with_weights(
            Tensor<scalar_t, 3> output,
            Tensor<scalar_t, 2> tensor_a,
            Tensor<scalar_t, 2> tensor_r,
            Tensor<scalar_t, 3> tensor_x,
            Tensor<scalar_t, 1> tensor_c,
            Tensor<int, 1> tensor_i,
            Tensor<int, 1> tensor_j,
            Tensor<int, 1> tensor_m_1,
            Tensor<int, 1> tensor_m_2,
            Tensor<int, 1> tensor_m_3
        );
    }
}


#endif
