#include "cpu.tpp"

// explicit instanciations of templates
template void mops::outer_product_scatter_add<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

template void mops::outer_product_scatter_add<double>(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

template void mops::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

template void mops::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

#ifndef MOPS_CUDA_ENABLED
template <typename scalar_t>
void mops::cuda::
    outer_product_scatter_add(Tensor<scalar_t, 3>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<int32_t, 1>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

// explicit instanciations of CUDA templates
template void mops::cuda::outer_product_scatter_add<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B, Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add<double>(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B, Tensor<int32_t, 1> indices_output
);

template <typename scalar_t>
void mops::cuda::outer_product_scatter_add_vjp(
    Tensor<scalar_t, 2> grad_A,
    Tensor<scalar_t, 2> grad_B,
    Tensor<scalar_t, 3> grad_output,
    Tensor<scalar_t, 2> A,
    Tensor<scalar_t, 2> B,
    Tensor<int32_t, 1> indices_output
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template void mops::cuda::outer_product_scatter_add_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::outer_product_scatter_add_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<int32_t, 1> indices_output
);

#endif
