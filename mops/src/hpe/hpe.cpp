#include "cpu.tpp"

// explicit instanciations of templates
template void mops::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A);

template void mops::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::homogeneous_polynomial_evaluation_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::homogeneous_polynomial_evaluation_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);


#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::homogeneous_polynomial_evaluation(
    Tensor<scalar_t, 1> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 1> C,
    Tensor<int32_t, 2> indices_A) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A);

template void mops::cuda::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A);
