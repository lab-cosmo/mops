#include "mops/torch/opsa.hpp"
#include "mops/torch/utils.hpp"

using namespace mops_torch;

torch::Tensor
mops_torch::outer_product_scatter_add(torch::Tensor A, torch::Tensor B,
                                      torch::Tensor indices_output,
                                      int64_t output_size) {
    return OuterProductScatterAdd::apply(A, B, indices_output, output_size);
}

torch::Tensor OuterProductScatterAdd::forward(
    torch::autograd::AutogradContext *ctx, torch::Tensor A, torch::Tensor B,
    torch::Tensor indices_output, int64_t output_size) {
    check_all_same_device({A, B, indices_output});
    check_floating_dtype({A, B});
    check_integer_dtype({indices_output});
    check_number_of_dimensions(A, 2, "A", "outer_product_scatter_add");
    check_number_of_dimensions(B, 2, "B", "outer_product_scatter_add");
    check_number_of_dimensions(indices_output, 1, "indices_output",
                               "outer_product_scatter_add");
    // Shape consistency checks are performed inside
    // mops::outer_product_scatter_add

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::zeros(
            {output_size, A.size(1), B.size(1)},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device()));

        assert(output.is_contiguous());

        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(), "outer_product_scatter_add", [&]() {
                mops::outer_product_scatter_add<scalar_t>(
                    torch_to_mops_2d<scalar_t>(
                        output.reshape({-1, output.size(1) * output.size(2)})),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });
    } else {

#ifndef MOPS_CUDA_ENABLED
        C10_THROW_ERROR(ValueError, "MOPS was not compiled with CUDA support " +
                                        A.device().str());
#else
        output = torch::empty(
            {output_size, A.size(1), B.size(1)},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device()));

        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(), "outer_product_scatter_add", [&]() {
                mops::cuda::outer_product_scatter_add<scalar_t>(
                    torch_to_mops_2d<scalar_t>(
                        output.reshape({-1, output.size(1) * output.size(2)})),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });

#endif
    }

    if (A.requires_grad() || B.requires_grad()) {
        ctx->save_for_backward({A, B, indices_output});
    }

    return {output};
}

std::vector<torch::Tensor>
OuterProductScatterAdd::backward(torch::autograd::AutogradContext *ctx,
                                 std::vector<torch::Tensor> grad_outputs) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto B = saved_variables[1];
    auto indices_output = saved_variables[2];

    auto grad_output = grad_outputs[0];

    if (!grad_output.is_contiguous()) {
        throw std::runtime_error("expected contiguous grad_output");
    }

    auto grad_A = torch::Tensor();
    auto grad_B = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(), "outer_product_scatter_add_vjp", [&]() {
                auto mops_grad_A = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (A.requires_grad()) {
                    grad_A = torch::zeros_like(A);
                    mops_grad_A = torch_to_mops_2d<scalar_t>(grad_A);
                }

                auto mops_grad_B = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (B.requires_grad()) {
                    grad_B = torch::zeros_like(B);
                    mops_grad_B = torch_to_mops_2d<scalar_t>(grad_B);
                }

                mops::outer_product_scatter_add_vjp<scalar_t>(
                    mops_grad_A, mops_grad_B,
                    torch_to_mops_2d<scalar_t>(grad_output.reshape(
                        {-1, grad_output.size(1) * grad_output.size(2)})),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });
    } else {

#ifndef MOPS_CUDA_ENABLED
        C10_THROW_ERROR(ValueError, "MOPS was not compiled with CUDA support " +
                                        A.device().str());
#elseValueError, "MOPS was not compiled with CUDA support " +
                                        auto mops_grad_A = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (A.requires_grad()) {
                    grad_A = torch::empty_like(A);
                    mops_grad_A = torch_to_mops_2d<scalar_t>(grad_A);
                }

                auto mops_grad_B = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (B.requires_grad()) {
                    grad_B = torch::empty_like(B);
                    mops_grad_B = torch_to_mops_2d<scalar_t>(grad_B);
                }

                mops::cuda::outer_product_scatter_add_vjp<scalar_t>(
                    mops_grad_A, mops_grad_B,
                    torch_to_mops_2d<scalar_t>(grad_output.reshape(
                        {-1, grad_output.size(1) * grad_output.size(2)})),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });
#endif
    }

    return {grad_A, grad_B, torch::Tensor(), torch::Tensor()};
}
