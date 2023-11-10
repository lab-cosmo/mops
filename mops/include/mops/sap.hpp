#ifndef MOPS_SAP_HPP
#define MOPS_SAP_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
    /// TODO
    template<typename scalar_t>
    void MOPS_EXPORT sparse_accumulation_of_products(
        Tensor<scalar_t, 2> output,
        Tensor<scalar_t, 2> tensor_a,
        Tensor<scalar_t, 2> tensor_b,
        Tensor<scalar_t, 1> tensor_c,
        Tensor<int32_t, 1> p_a,
        Tensor<int32_t, 1> p_b,
        Tensor<int32_t, 1> p_o
    );

    // these templates will be precompiled and provided in the mops library
    extern template void sparse_accumulation_of_products(
        Tensor<float, 2> output,
        Tensor<float, 2> tensor_a,
        Tensor<float, 2> tensor_b,
        Tensor<float, 1> tensor_c,
        Tensor<int32_t, 1> p_a,
        Tensor<int32_t, 1> p_b,
        Tensor<int32_t, 1> p_o
    );

    extern template void sparse_accumulation_of_products(
        Tensor<double, 2> output,
        Tensor<double, 2> tensor_a,
        Tensor<double, 2> tensor_b,
        Tensor<double, 1> tensor_c,
        Tensor<int32_t, 1> p_a,
        Tensor<int32_t, 1> p_b,
        Tensor<int32_t, 1> p_o
    );

    namespace cuda {
        /// CUDA version of mops::sparse_accumulation_of_products
        template<typename scalar_t>
        void MOPS_EXPORT sparse_accumulation_of_products(
            Tensor<scalar_t, 2> output,
            Tensor<scalar_t, 2> tensor_a,
            Tensor<scalar_t, 2> tensor_b,
            Tensor<scalar_t, 1> tensor_c,
            Tensor<int32_t, 1> p_a,
            Tensor<int32_t, 1> p_b,
            Tensor<int32_t, 1> p_o
        );
    }
}


#endif
