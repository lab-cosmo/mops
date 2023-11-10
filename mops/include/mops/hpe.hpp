#ifndef MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP
#define MOPS_OUTER_PRODUCT_SCATTER_ADD_HPP

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {
    /// TODO
    template<typename scalar_t>
    void MOPS_EXPORT homogeneous_polynomial_evaluation(
        Tensor<scalar_t, 1> output,
        Tensor<scalar_t, 2> tensor_a,
        Tensor<scalar_t, 1> tensor_c,
        Tensor<int32_t, 2> p
    );

    // these templates will be precompiled and provided in the mops library
    extern template void homogeneous_polynomial_evaluation(
        Tensor<float, 1> output,
        Tensor<float, 2> tensor_a,
        Tensor<float, 1> tensor_c,
        Tensor<int32_t, 2> p
    );

    extern template void homogeneous_polynomial_evaluation(
        Tensor<double, 1> output,
        Tensor<double, 2> tensor_a,
        Tensor<double, 1> tensor_c,
        Tensor<int32_t, 2> p
    );

    namespace cuda {
        /// CUDA version of mops::homogeneous_polynomial_evaluation
        template<typename scalar_t>
        void MOPS_EXPORT homogeneous_polynomial_evaluation(
            Tensor<scalar_t, 1> output,
            Tensor<scalar_t, 2> tensor_a,
            Tensor<scalar_t, 1> tensor_c,
            Tensor<int32_t, 2> p
        );
    }
}


#endif
