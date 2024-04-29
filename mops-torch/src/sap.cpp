#include "mops/torch/sap.hpp"
#include "mops/torch/utils.hpp"

using namespace mops_torch;

torch::Tensor mops_torch::sparse_accumulation_of_products(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor indices_A,
    torch::Tensor indices_B,
    torch::Tensor indices_output,
    int64_t output_size
) {
    return SparseAccumulationOfProducts::apply(
        A, B, C, indices_A, indices_B, indices_output, output_size
    );
}

torch::Tensor SparseAccumulationOfProducts::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor indices_A,
    torch::Tensor indices_B,
    torch::Tensor indices_output,
    int64_t output_size
) {
    details::check_all_same_device({A, B, C, indices_A, indices_B, indices_output});
    details::check_floating_dtype({A, B, C});
    details::check_integer_dtype({indices_A, indices_B, indices_output});
    details::check_number_of_dimensions(A, 2, "A", "sparse_accumulation_of_products");
    details::check_number_of_dimensions(B, 2, "B", "sparse_accumulation_of_products");
    details::check_number_of_dimensions(indices_A, 1, "indices_A", "sparse_accumulation_of_products");
    details::check_number_of_dimensions(indices_B, 1, "indices_B", "sparse_accumulation_of_products");
    details::check_number_of_dimensions(
        indices_output, 1, "indices_output", "sparse_accumulation_of_products"
    );
    // Shape consistency checks are performed inside
    // mops::sparse_accumulation_of_products

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::empty(
            {A.size(0), output_size},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device())
        );

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products", [&]() {
            mops::sparse_accumulation_of_products<scalar_t>(
                details::torch_to_mops_2d<scalar_t>(output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_1d<int32_t>(indices_A),
                details::torch_to_mops_1d<int32_t>(indices_B),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else if (A.device().is_cuda()) {
        output = torch::empty(
            {A.size(0), output_size},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device())
        );

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products", [&]() {
            mops::cuda::sparse_accumulation_of_products<scalar_t>(
                details::torch_to_mops_2d<scalar_t>(output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_1d<int32_t>(indices_A),
                details::torch_to_mops_1d<int32_t>(indices_B),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "sparse_accumulation_of_products is not implemented for device " + A.device().str()
        );
    }

    if (A.requires_grad() || B.requires_grad()) {
        ctx->save_for_backward({A, B, C, indices_A, indices_B, indices_output});
    }

    return {output};
}

std::vector<torch::Tensor> SparseAccumulationOfProducts::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto B = saved_variables[1];
    auto C = saved_variables[2];
    auto indices_A = saved_variables[3];
    auto indices_B = saved_variables[4];
    auto indices_output = saved_variables[5];

    auto grad_output = grad_outputs[0].contiguous();

    auto results = SparseAccumulationOfProductsBackward::apply(
        grad_output, A, B, C, indices_A, indices_B, indices_output
    );
    auto grad_A = results[0];
    if (grad_output.requires_grad() || A.requires_grad() || B.requires_grad()) {
        ctx->save_for_backward({grad_output, A, B, C, indices_A, indices_B, indices_output});
    }
    auto grad_B = results[1];

    return {
        grad_A,
        grad_B,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()
    };
}

std::vector<torch::Tensor> SparseAccumulationOfProductsBackward::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor grad_output,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor indices_A,
    torch::Tensor indices_B,
    torch::Tensor indices_output
) {
    if (C.requires_grad()) {
        C10_THROW_ERROR(
            ValueError,
            "gradients with respect to C are not supported "
            "in sparse_accumulation_of_products"
        );
    }

    auto grad_A = torch::Tensor();
    auto grad_B = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products_vjp", [&]() {
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

            mops::sparse_accumulation_of_products_vjp<scalar_t>(
                mops_grad_A,
                mops_grad_B,
                details::torch_to_mops_2d<scalar_t>(grad_output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_1d<int32_t>(indices_A),
                details::torch_to_mops_1d<int32_t>(indices_B),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else if (A.device().is_cuda()) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products_vjp", [&]() {
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

            mops::cuda::sparse_accumulation_of_products_vjp<scalar_t>(
                mops_grad_A,
                mops_grad_B,
                details::torch_to_mops_2d<scalar_t>(grad_output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_1d<int32_t>(indices_A),
                details::torch_to_mops_1d<int32_t>(indices_B),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "sparse_accumulation_of_products is not implemented for device " + A.device().str()
        );
    }

    if (grad_output.requires_grad() || A.requires_grad() || B.requires_grad()) {
        ctx->save_for_backward({grad_output, A, B, C, indices_A, indices_B, indices_output});
    }

    return {grad_A, grad_B};
}

std::vector<torch::Tensor> SparseAccumulationOfProductsBackward::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto grad_output = saved_variables[0];
    auto A = saved_variables[1];
    auto B = saved_variables[2];
    auto C = saved_variables[3];
    auto indices_A = saved_variables[4];
    auto indices_B = saved_variables[5];
    auto indices_output = saved_variables[6];
    auto grad_grad_A = grad_outputs[0].contiguous();
    auto grad_grad_B = grad_outputs[1].contiguous();

    auto grad_grad_output = torch::Tensor();
    auto grad_A_2 = torch::Tensor();
    auto grad_B_2 = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products_vjp_vjp", [&]() {
            auto mops_grad_grad_output = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (grad_output.requires_grad()) {
                grad_grad_output = torch::empty_like(grad_output);
                mops_grad_grad_output = details::torch_to_mops_2d<scalar_t>(grad_grad_output);
            }

            auto mops_grad_A_2 = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (A.requires_grad()) {
                grad_A_2 = torch::empty_like(A);
                mops_grad_A_2 = details::torch_to_mops_2d<scalar_t>(grad_A_2);
            }

            auto mops_grad_B_2 = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (B.requires_grad()) {
                grad_B_2 = torch::empty_like(B);
                mops_grad_B_2 = details::torch_to_mops_2d<scalar_t>(grad_B_2);
            }

            auto mops_grad_grad_A = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (grad_grad_A.defined()) {
                mops_grad_grad_A = details::torch_to_mops_2d<scalar_t>(grad_grad_A);
            }

            auto mops_grad_grad_B = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (grad_grad_B.defined()) {
                mops_grad_grad_B = details::torch_to_mops_2d<scalar_t>(grad_grad_B);
            }

            mops::sparse_accumulation_of_products_vjp_vjp<scalar_t>(
                mops_grad_grad_output,
                mops_grad_A_2,
                mops_grad_B_2,
                mops_grad_grad_A,
                mops_grad_grad_B,
                details::torch_to_mops_2d<scalar_t>(grad_output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_2d<scalar_t>(B),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_1d<int32_t>(indices_A),
                details::torch_to_mops_1d<int32_t>(indices_B),
                details::torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else if (A.device().is_cuda()) {
        C10_THROW_ERROR(
            ValueError, "sparse_accumulation_of_products_vjp_vjp is not implemented for CUDA yet"
        );
    } else {
        C10_THROW_ERROR(
            ValueError,
            "sparse_accumulation_of_products_vjp_vjp is not implemented for device " +
                A.device().str()
        );
    }

    return {
        grad_grad_output,
        grad_A_2,
        grad_B_2,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor()
    };
}
