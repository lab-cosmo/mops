#include "mops/torch/sasaw.hpp"
#include "mops/torch/utils.hpp"

using namespace mops_torch;

torch::Tensor mops_torch::sparse_accumulation_scatter_add_with_weights(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor W,
    torch::Tensor indices_A,
    torch::Tensor indices_W_1,
    torch::Tensor indices_W_2,
    torch::Tensor indices_output_1,
    torch::Tensor indices_output_2,
    int64_t output_size
) {
    return SparseAccumulationScatterAddWithWeights::apply(
        A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2, output_size
    );
}

torch::Tensor SparseAccumulationScatterAddWithWeights::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor W,
    torch::Tensor indices_A,
    torch::Tensor indices_W_1,
    torch::Tensor indices_W_2,
    torch::Tensor indices_output_1,
    torch::Tensor indices_output_2,
    int64_t output_size
) {
    details::check_all_same_device(
        {A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2}
    );
    details::check_floating_dtype({A, B, C, W});
    details::check_integer_dtype(
        {indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2}
    );
    details::check_number_of_dimensions(A, 2, "A", "sparse_accumulation_scatter_add_with_weights");
    details::check_number_of_dimensions(B, 2, "B", "sparse_accumulation_scatter_add_with_weights");
    details::check_number_of_dimensions(W, 3, "W", "sparse_accumulation_scatter_add_with_weights");
    details::check_number_of_dimensions(
        indices_A, 1, "indices_A", "sparse_accumulation_scatter_add_with_weights"
    );
    details::check_number_of_dimensions(
        indices_W_1, 1, "indices_W_1", "sparse_accumulation_scatter_add_with_weights"
    );
    details::check_number_of_dimensions(
        indices_W_2, 1, "indices_W_2", "sparse_accumulation_scatter_add_with_weights"
    );
    details::check_number_of_dimensions(
        indices_output_1, 1, "indices_output_1", "sparse_accumulation_scatter_add_with_weights"
    );
    details::check_number_of_dimensions(
        indices_output_2, 1, "indices_output_2", "sparse_accumulation_scatter_add_with_weights"
    );
    // Shape consistency checks are performed inside
    // mops::sparse_accumulation_scatter_add_with_weights

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::empty(
            {W.size(0), output_size, W.size(2)},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device())
        );
        assert(output.is_contiguous());

        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(),
            "sparse_accumulation_scatter_add_with_weights",
            [&]() {
                mops::sparse_accumulation_scatter_add_with_weights<scalar_t>(
                    details::torch_to_mops_3d<scalar_t>(output),
                    details::torch_to_mops_2d<scalar_t>(A),
                    details::torch_to_mops_2d<scalar_t>(B),
                    details::torch_to_mops_1d<scalar_t>(C),
                    details::torch_to_mops_3d<scalar_t>(W),
                    details::torch_to_mops_1d<int32_t>(indices_A),
                    details::torch_to_mops_1d<int32_t>(indices_W_1),
                    details::torch_to_mops_1d<int32_t>(indices_W_2),
                    details::torch_to_mops_1d<int32_t>(indices_output_1),
                    details::torch_to_mops_1d<int32_t>(indices_output_2)
                );
            }
        );
    } else {
        C10_THROW_ERROR(
            ValueError,
            "sparse_accumulation_scatter_add_with_"
            "weights is not implemented for device " +
                A.device().str()
        );
    }

    if (A.requires_grad() || B.requires_grad() || W.requires_grad()) {
        ctx->save_for_backward(
            {A, B, C, W, indices_A, indices_W_1, indices_W_2, indices_output_1, indices_output_2}
        );
    }

    return {output};
}

std::vector<torch::Tensor> SparseAccumulationScatterAddWithWeights::backward(
    torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto B = saved_variables[1];
    auto C = saved_variables[2];
    auto W = saved_variables[3];
    auto indices_A = saved_variables[4];
    auto indices_W_1 = saved_variables[5];
    auto indices_W_2 = saved_variables[6];
    auto indices_output_1 = saved_variables[7];
    auto indices_output_2 = saved_variables[8];

    auto grad_output = grad_outputs[0].contiguous();

    if (C.requires_grad()) {
        C10_THROW_ERROR(
            ValueError,
            "gradients with respect to C are not supported "
            "in sparse_accumulation_scatter_add_with_weights"
        );
    }

    auto grad_A = torch::Tensor();
    auto grad_B = torch::Tensor();
    auto grad_W = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(),
            "sparse_accumulation_scatter_add_with_weights_vjp",
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

                auto mops_grad_W = mops::Tensor<scalar_t, 3>{nullptr, {0, 0, 0}};
                if (W.requires_grad()) {
                    grad_W = torch::empty_like(W);
                    mops_grad_W = details::torch_to_mops_3d<scalar_t>(grad_W);
                }

                mops::sparse_accumulation_scatter_add_with_weights_vjp<scalar_t>(
                    mops_grad_A,
                    mops_grad_B,
                    mops_grad_W,
                    details::torch_to_mops_3d<scalar_t>(grad_output),
                    details::torch_to_mops_2d<scalar_t>(A),
                    details::torch_to_mops_2d<scalar_t>(B),
                    details::torch_to_mops_1d<scalar_t>(C),
                    details::torch_to_mops_3d<scalar_t>(W),
                    details::torch_to_mops_1d<int32_t>(indices_A),
                    details::torch_to_mops_1d<int32_t>(indices_W_1),
                    details::torch_to_mops_1d<int32_t>(indices_W_2),
                    details::torch_to_mops_1d<int32_t>(indices_output_1),
                    details::torch_to_mops_1d<int32_t>(indices_output_2)
                );
            }
        );
    } else {
        C10_THROW_ERROR(
            ValueError,
            "sparse_accumulation_scatter_add_with_"
            "weights is not implemented for device " +
                A.device().str()
        );
    }

    return {
        grad_A,
        grad_B,
        torch::Tensor(),
        grad_W,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()
    };
}
