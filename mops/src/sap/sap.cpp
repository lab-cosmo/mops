#include "cpu.tpp"

// explicit instantiations of templates
template void mops::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products_vjp_vjp<float>(
    Tensor<float, 2> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::sparse_accumulation_of_products_vjp_vjp<double>(
    Tensor<double, 2> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

#ifndef MOPS_CUDA_ENABLED
template <typename scalar_t>
void mops::cuda::
    sparse_accumulation_of_products(Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::
    sparse_accumulation_of_products_vjp(Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::sparse_accumulation_of_products_vjp_vjp(
    Tensor<scalar_t, 2> /*grad_grad_output*/,
    Tensor<scalar_t, 2> /*grad_A_2*/,
    Tensor<scalar_t, 2> /*grad_B_2*/,
    Tensor<scalar_t, 2> /*grad_grad_A*/,
    Tensor<scalar_t, 2> /*grad_grad_B*/,
    Tensor<scalar_t, 2> /*grad_output*/,
    Tensor<scalar_t, 2> /*A*/,
    Tensor<scalar_t, 2> /*B*/,
    Tensor<scalar_t, 1> /*C*/,
    Tensor<int32_t, 1> /*indices_A*/,
    Tensor<int32_t, 1> /*indices_B*/,
    Tensor<int32_t, 1> /*indices_output*/
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

// explicit instantiations of CUDA templates
template void mops::cuda::sparse_accumulation_of_products<float>(
    Tensor<float, 2> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products<double>(
    Tensor<double, 2> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products_vjp_vjp<float>(
    Tensor<float, 2> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 2> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

template void mops::cuda::sparse_accumulation_of_products_vjp_vjp<double>(
    Tensor<double, 2> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 2> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_B,
    Tensor<int32_t, 1> indices_output
);

#endif
