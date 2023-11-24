#include "cpu.tpp"

// explicit instanciations of templates
template void mops::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<float, 1> C, Tensor<float, 3> W, Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1, Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1, Tensor<int, 1> indices_output_2);

template void mops::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1,
    Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1,
    Tensor<int, 1> indices_output_2
);

template void mops::sparse_accumulation_scatter_add_with_weights_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1,
    Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1,
    Tensor<int, 1> indices_output_2
);

template void mops::sparse_accumulation_scatter_add_with_weights_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1,
    Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1,
    Tensor<int, 1> indices_output_2
);


#ifdef MOPS_CUDA_ENABLED
#include "cuda.tpp"
#else
template <typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights(
    Tensor<scalar_t, 3> output, Tensor<scalar_t, 2> A, Tensor<scalar_t, 2> B,
    Tensor<scalar_t, 1> C, Tensor<scalar_t, 3> W, Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1, Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1, Tensor<int, 1> indices_output_2) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

#endif

// explicit instanciations of CUDA templates
template void mops::cuda::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output, Tensor<float, 2> A, Tensor<float, 2> B,
    Tensor<float, 1> C, Tensor<float, 3> W, Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1, Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1, Tensor<int, 1> indices_output_2);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output, Tensor<double, 2> A, Tensor<double, 2> B,
    Tensor<double, 1> C, Tensor<double, 3> W, Tensor<int, 1> indices_A,
    Tensor<int, 1> indices_W_1, Tensor<int, 1> indices_W_2,
    Tensor<int, 1> indices_output_1, Tensor<int, 1> indices_output_2);
