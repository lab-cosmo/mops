#include <stdexcept>

#include "mops/sasaw.hpp"

using namespace mops;
using namespace mops::cuda;
using namespace std;

template <typename scalar_t>
void mops::cuda::
    sparse_accumulation_scatter_add_with_weights(Tensor<scalar_t, 3>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<scalar_t, 3>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, Tensor<int32_t, 1>, void*) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::
    sparse_accumulation_scatter_add_with_weights_vjp(Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 3>, Tensor<scalar_t, 3>, Tensor<scalar_t, 2>, Tensor<scalar_t, 2>, Tensor<scalar_t, 1>, Tensor<scalar_t, 3>, Tensor<int, 1>, Tensor<int, 1>, Tensor<int, 1>, Tensor<int, 1>, Tensor<int, 1>, void*) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

template <typename scalar_t>
void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp_vjp(
    Tensor<scalar_t, 3> /*grad_grad_output*/,
    Tensor<scalar_t, 2> /*grad_A_2*/,
    Tensor<scalar_t, 2> /*grad_B_2*/,
    Tensor<scalar_t, 3> /*grad_W_2*/,
    Tensor<scalar_t, 2> /*grad_grad_A*/,
    Tensor<scalar_t, 2> /*grad_grad_B*/,
    Tensor<scalar_t, 3> /*grad_grad_W*/,
    Tensor<scalar_t, 3> /*grad_output*/,
    Tensor<scalar_t, 2> /*A*/,
    Tensor<scalar_t, 2> /*B*/,
    Tensor<scalar_t, 1> /*C*/,
    Tensor<scalar_t, 3> /*W*/,
    Tensor<int32_t, 1> /*indices_A*/,
    Tensor<int32_t, 1> /*indices_W_1*/,
    Tensor<int32_t, 1> /*indices_W_2*/,
    Tensor<int32_t, 1> /*indices_output_1*/,
    Tensor<int32_t, 1> /*indices_output_2*/,
    void* /*cuda_stream*/
) {
    throw std::runtime_error("MOPS was not compiled with CUDA support");
}

// explicit instantiations of CUDA templates
template void mops::cuda::sparse_accumulation_scatter_add_with_weights<float>(
    Tensor<float, 3> output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights<double>(
    Tensor<double, 3> output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp<float>(
    Tensor<float, 2> grad_A,
    Tensor<float, 2> grad_B,
    Tensor<float, 3> grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp<double>(
    Tensor<double, 2> grad_A,
    Tensor<double, 2> grad_B,
    Tensor<double, 3> grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp_vjp<float>(
    Tensor<float, 3> grad_grad_output,
    Tensor<float, 2> grad_A_2,
    Tensor<float, 2> grad_B_2,
    Tensor<float, 3> grad_W_2,
    Tensor<float, 2> grad_grad_A,
    Tensor<float, 2> grad_grad_B,
    Tensor<float, 3> grad_grad_W,
    Tensor<float, 3> grad_output,
    Tensor<float, 2> A,
    Tensor<float, 2> B,
    Tensor<float, 1> C,
    Tensor<float, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);

template void mops::cuda::sparse_accumulation_scatter_add_with_weights_vjp_vjp<double>(
    Tensor<double, 3> grad_grad_output,
    Tensor<double, 2> grad_A_2,
    Tensor<double, 2> grad_B_2,
    Tensor<double, 3> grad_W_2,
    Tensor<double, 2> grad_grad_A,
    Tensor<double, 2> grad_grad_B,
    Tensor<double, 3> grad_grad_W,
    Tensor<double, 3> grad_output,
    Tensor<double, 2> A,
    Tensor<double, 2> B,
    Tensor<double, 1> C,
    Tensor<double, 3> W,
    Tensor<int32_t, 1> indices_A,
    Tensor<int32_t, 1> indices_W_1,
    Tensor<int32_t, 1> indices_W_2,
    Tensor<int32_t, 1> indices_output_1,
    Tensor<int32_t, 1> indices_output_2,
    void* cuda_stream
);