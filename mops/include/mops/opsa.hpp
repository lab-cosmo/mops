#ifndef MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP
#define MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"

namespace mops {
    /// TODO
    template<typename scalar_t>
    void MOPS_EXPORT outer_product_scatter_add(
        scalar_t* output,
        size_t output_shape_1,
        size_t output_shape_2,
        const scalar_t* tensor_a,
        size_t tensor_a_shape_1,
        size_t tensor_a_shape_2,
        const scalar_t* tensor_b,
        size_t tensor_b_shape_1,
        size_t tensor_b_shape_2,
        const int32_t* indexes,
        size_t indexes_shape_1
    );

    // these templates will be precompiled and provided in the mops library
    extern template void outer_product_scatter_add<float>(
        float* output,
        size_t output_shape_1,
        size_t output_shape_2,
        const float* tensor_a,
        size_t tensor_a_shape_1,
        size_t tensor_a_shape_2,
        const float* tensor_b,
        size_t tensor_b_shape_1,
        size_t tensor_b_shape_2,
        const int32_t* indexes,
        size_t indexes_shape_1
    );

    extern template void outer_product_scatter_add<double>(
        double* output,
        size_t output_shape_1,
        size_t output_shape_2,
        const double* tensor_a,
        size_t tensor_a_shape_1,
        size_t tensor_a_shape_2,
        const double* tensor_b,
        size_t tensor_b_shape_1,
        size_t tensor_b_shape_2,
        const int32_t* indexes,
        size_t indexes_shape_1
    );


    namespace cuda {
        /// CUDA version of mops::outer_product_scatter_add
        template<typename scalar_t>
        void MOPS_EXPORT outer_product_scatter_add(
            scalar_t* output,
            size_t output_shape_1,
            size_t output_shape_2,
            const scalar_t* tensor_a,
            size_t tensor_a_shape_1,
            size_t tensor_a_shape_2,
            const scalar_t* tensor_b,
            size_t tensor_b_shape_1,
            size_t tensor_b_shape_2,
            const int32_t* indexes,
            size_t indexes_shape_1
        );
    }
}


#endif
