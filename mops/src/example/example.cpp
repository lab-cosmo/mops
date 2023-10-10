#include "cpu.tpp"


// explicit instanciations
template void mops::example_values<float>(
    const float* input,
    size_t input_length,
    float* output,
    size_t output_length
);

template void mops::example_values<double>(
    const double* input,
    size_t input_length,
    double* output,
    size_t output_length
);

template void mops::example_jvp<float>(
    const float* grad_output,
    size_t grad_output_length,
    float* grad_input,
    size_t grad_input_length
);

template void mops::example_jvp<double>(
    const double* grad_output,
    size_t grad_output_length,
    double* grad_input,
    size_t grad_input_length
);



#ifdef MOPS_CUDA_ENABLED

#include "cuda.tpp"

#else

template<typename scalar_t>
void mops::cuda::example_values(
    int cuda_device,
    const scalar_t* input,
    size_t input_length,
    scalar_t* output,
    size_t output_length
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template<typename scalar_t>
void mops::cuda::example_jvp(
    int cuda_device,
    const scalar_t* grad_output,
    size_t grad_output_length,
    scalar_t* grad_input,
    size_t grad_input_length
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations
template void mops::cuda::example_values<float>(
    int cuda_device,
    const float* input,
    size_t input_length,
    float* output,
    size_t output_length
);

template void mops::cuda::example_values<double>(
    int cuda_device,
    const double* input,
    size_t input_length,
    double* output,
    size_t output_length
);

template void mops::cuda::example_jvp<float>(
    int cuda_device,
    const float* grad_output,
    size_t grad_output_length,
    float* grad_input,
    size_t grad_input_length
);

template void mops::cuda::example_jvp<double>(
    int cuda_device,
    const double* grad_output,
    size_t grad_output_length,
    double* grad_input,
    size_t grad_input_length
);
