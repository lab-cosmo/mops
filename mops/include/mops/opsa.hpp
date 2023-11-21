#ifndef MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP
#define MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
    /// TODO
    template<typename scalar_t>
    void MOPS_EXPORT outer_product_scatter_add(
        Tensor<scalar_t, 2> output,
        Tensor<scalar_t, 2> A,
        Tensor<scalar_t, 2> B,
        Tensor<int32_t, 1> indices_output
    );

    // these templates will be precompiled and provided in the mops library
    extern template void outer_product_scatter_add(
        Tensor<float, 2> output,
        Tensor<float, 2> A,
        Tensor<float, 2> B,
        Tensor<int32_t, 1> indices_output
    );

    extern template void outer_product_scatter_add(
        Tensor<double, 2> output,
        Tensor<double, 2> A,
        Tensor<double, 2> B,
        Tensor<int32_t, 1> indices_output
    );

    namespace cuda {
        /// CUDA version of mops::outer_product_scatter_add
        template<typename scalar_t>
        void MOPS_EXPORT outer_product_scatter_add(
            Tensor<scalar_t, 2> output,
            Tensor<scalar_t, 2> A,
            Tensor<scalar_t, 2> B,
            Tensor<int32_t, 1> indices_output
        );
    }
}


#endif
