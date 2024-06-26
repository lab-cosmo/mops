#include "mops/torch/opsaw.hpp"
#include "mops/torch/utils.hpp"

using namespace mops_torch;

torch::Tensor mops_torch::outer_product_scatter_add_with_weights(
    torch::Tensor A, torch::Tensor B, torch::Tensor W, torch::Tensor indices_W, torch::Tensor indices_output
) {
    return OuterProductScatterAddWithWeights::apply(A, B, W, indices_W, indices_output);
}

torch::Tensor OuterProductScatterAddWithWeights::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor W,
    torch::Tensor indices_W,
    torch::Tensor indices_output
) {
    details::check_all_same_device({A, B, W, indices_W, indices_output});
    details::check_floating_dtype({A, B, W});
    details::check_integer_dtype({indices_W, indices_output});
    details::check_number_of_dimensions(A, 2, "A", "outer_product_scatter_add_with_weights");
    details::check_number_of_dimensions(B, 2, "B", "outer_product_scatter_add_with_weights");
    details::check_number_of_dimensions(W, 2, "W", "outer_product_scatter_add_with_weights");
    details::check_number_of_dimensions(
        indices_W, 1, "indices_W", "outer_product_scatter_add_with_weights"
    );
    details::check_number_of_dimensions(
        indices_output, 1, "indices_output", "outer_product_scatter_add_with_weights"
    );
    // Shape consistency checks are performed inside
    // mops::outer_product_scatter_add_with_weights

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::empty(
            {W.size(0), A.size(1), B.size(1)},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device())
        );
        assert(output.is_contiguous());

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "outer_product_scatter_add_with_weights", [&]() {
            mops::outer_product_scatter_add_with_weights<scalar_t>(
                details::torch_to_mops_3d<scalar_t>(output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_2d<scalar_t>(W),
                details::torch_to_mops_1d<int32_t>(indices_W),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "outer_product_scatter_add_with_weights is "
            "not implemented for device " +
                A.device().str()
        );
    }

    if (A.requires_grad() || B.requires_grad() || W.requires_grad()) {
        ctx->save_for_backward({A, B, W, indices_W, indices_output});
    }

    return {output};
}

std::vector<torch::Tensor> OuterProductScatterAddWithWeights::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto B = saved_variables[1];
    auto W = saved_variables[2];
    auto indices_W = saved_variables[3];
    auto indices_output = saved_variables[4];

    auto grad_output = grad_outputs[0].contiguous();

    auto results = OuterProductScatterAddWithWeightsBackward::apply(
        grad_output, A, B, W, indices_W, indices_output
    );
    auto grad_A = results[0];
    auto grad_B = results[1];
    auto grad_W = results[2];

    return {grad_A, grad_B, grad_W, torch::Tensor(), torch::Tensor()};
}

std::vector<torch::Tensor> OuterProductScatterAddWithWeightsBackward::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor grad_output,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor W,
    torch::Tensor indices_W,
    torch::Tensor indices_output
) {
    auto grad_A = torch::Tensor();
    auto grad_B = torch::Tensor();
    auto grad_W = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(),
            "outer_product_scatter_add_with_weights_vjp",
            [&]() {
                auto mops_grad_A = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (A.requires_grad()) {
                    grad_A = torch::empty_like(A);
                    mops_grad_A = details::torch_to_mops_2d<scalar_t>(grad_A);
                }

                auto mops_grad_B = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (B.requires_grad()) {
                    grad_B = torch::empty_like(B);
                    mops_grad_B = details::torch_to_mops_2d<scalar_t>(grad_B);
                }

                auto mops_grad_W = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
                if (W.requires_grad()) {
                    grad_W = torch::empty_like(W);
                    mops_grad_W = details::torch_to_mops_2d<scalar_t>(grad_W);
                }

                mops::outer_product_scatter_add_with_weights_vjp<scalar_t>(
                    mops_grad_A,
                    mops_grad_B,
                    mops_grad_W,
                    details::torch_to_mops_3d<scalar_t>(grad_output),
                    details::torch_to_mops_2d<scalar_t>(A),
                    details::torch_to_mops_2d<scalar_t>(B),
                    details::torch_to_mops_2d<scalar_t>(W),
                    details::torch_to_mops_1d<int32_t>(indices_W),
                    details::torch_to_mops_1d<int32_t>(indices_output)
                );
            }
        );
    } else {
        C10_THROW_ERROR(
            ValueError,
            "outer_product_scatter_add_with_weights is "
            "not implemented for device " +
                A.device().str()
        );
    }

    return {grad_A, grad_B, grad_W};
}

std::vector<torch::Tensor> OuterProductScatterAddWithWeightsBackward::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {
    C10_THROW_ERROR(
        ValueError, "second derivatives are not supported in outer_product_scatter_add_with_weights"
    );
}
