#include "cpu.tpp"

// explicit instantiations of templates
template void mops::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C, Tensor<int32_t, 2> indices_A
);

template void mops::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A
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

template void mops::homogeneous_polynomial_evaluation_vjp_vjp<float>(
    Tensor<float, 1> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::homogeneous_polynomial_evaluation_vjp_vjp<double>(
    Tensor<double, 1> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

#ifndef MOPS_CUDA_ENABLED
template <typename scalar_t>
void mops::cuda::
    homogeneous_polynomial_evaluation(Tensor<scalar_t, 1>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<int32_t, 2>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::
    homogeneous_polynomial_evaluation_vjp(Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<int32_t, 2>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::
    homogeneous_polynomial_evaluation_vjp_vjp(Tensor<scalar_t, 1>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<int32_t, 2>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

// explicit instantiations of CUDA templates
template void mops::cuda::homogeneous_polynomial_evaluation<float>(
    Tensor<float, 1> output, Tensor<float, 2> A, Tensor<float, 1> C, Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation<double>(
    Tensor<double, 1> output, Tensor<double, 2> A, Tensor<double, 1> C, Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation_vjp_vjp<float>(
    Tensor<float, 1> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 1> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 1> C,
    Tensor<int32_t, 2> indices_A
);

template void mops::cuda::homogeneous_polynomial_evaluation_vjp_vjp<double>(
    Tensor<double, 1> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 1> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 1> C,
    Tensor<int32_t, 2> indices_A
);

#endif
