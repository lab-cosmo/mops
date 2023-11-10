#include "cpu.tpp"

// explicit instanciations of templates
template void mops::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output,
    Tensor<float, 2> tensor_a,
    Tensor<float, 1> tensor_c,
    Tensor<int32_t, 2> p
);

template void mops::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output,
    Tensor<double, 2> tensor_a,
    Tensor<double, 1> tensor_c,
    Tensor<int32_t, 2> p
);


#ifdef MOPS_CUDA_ENABLED
    #include "cuda.tpp"
#else
template<typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output,
    Tensor<scalar_t, 2> tensor_a,
    Tensor<scalar_t, 1> tensor_c,
    Tensor<int32_t, 2> p
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output,
    Tensor<float, 2> tensor_a,
    Tensor<float, 1> tensor_c,
    Tensor<int32_t, 2> p
);

template void mops::cuda::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output,
    Tensor<double, 2> tensor_a,
    Tensor<double, 1> tensor_c,
    Tensor<int32_t, 2> p
);
