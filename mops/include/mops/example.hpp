#ifndef MOPS_EXAMPLE_HPP
#define MOPS_EXAMPLE_HPP

#include <cstddef>

#include "mops/exports.h"

namespace mops {
    template<typename scalar_t>
    void example_values(
        const scalar_t* input,
        size_t input_length,
        scalar_t* output,
        size_t output_length
    );

    template<typename scalar_t>
    void example_jvp(
        const scalar_t* grad_output,
        size_t grad_output_length,
        scalar_t* grad_input,
        size_t grad_input_length
    );

    extern template void example_values<float>(
        const float* input,
        size_t input_length,
        float* output,
        size_t output_length
    );
    extern template void example_values<double>(
        const double* input,
        size_t input_length,
        double* output,
        size_t output_length
    );

    extern template void example_jvp<float>(
        const float* input,
        size_t input_length,
        float* output,
        size_t output_length
    );
    extern template void example_jvp<double>(
        const double* input,
        size_t input_length,
        double* output,
        size_t output_length
    );


    namespace cuda {
        template<typename scalar_t>
        void example_values(
            int cuda_device,
            const scalar_t* input,
            size_t input_length,
            scalar_t* output,
            size_t output_length
        );

        template<typename scalar_t>
        void example_jvp(
            int cuda_device,
            const scalar_t* grad_output,
            size_t grad_output_length,
            scalar_t* grad_input,
            size_t grad_input_length
        );

        extern template void example_values<float>(
            int cuda_device,
            const float* input,
            size_t input_length,
            float* output,
            size_t output_length
        );
        extern template void example_values<double>(
            int cuda_device,
            const double* input,
            size_t input_length,
            double* output,
            size_t output_length
        );

        extern template void example_jvp<float>(
            int cuda_device,
            const float* input,
            size_t input_length,
            float* output,
            size_t output_length
        );
        extern template void example_jvp<double>(
            int cuda_device,
            const double* input,
            size_t input_length,
            double* output,
            size_t output_length
        );
    }
}


#endif
