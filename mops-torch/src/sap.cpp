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
    return SparseAccumulationOfProducts::apply(A, B, C, indices_A, indices_B, indices_output, output_size)[0];
}

std::vector<torch::Tensor> SparseAccumulationOfProducts::forward(
    torch::autograd::AutogradContext *ctx,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor indices_A,
    torch::Tensor indices_B,
    torch::Tensor indices_output,
    int64_t output_size
) {
    // TODO: checks

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::zeros({A.size(0), output_size},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device())
        );
        assert(output.is_contiguous());

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products", [&](){
            mops::sparse_accumulation_of_products<scalar_t>(
                torch_to_mops_2d<scalar_t>(output),
                torch_to_mops_2d<scalar_t>(A),
                torch_to_mops_2d<scalar_t>(B),
                torch_to_mops_1d<scalar_t>(C),
                torch_to_mops_1d<int32_t>(indices_A),
                torch_to_mops_1d<int32_t>(indices_B),
                torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else {
        C10_THROW_ERROR(ValueError,
            "sparse_accumulation_of_products is not implemented for device " + A.device().str()
        );
    }

    if (A.requires_grad() || B.requires_grad()) {
        ctx->save_for_backward({A, B, C, indices_A, indices_B, indices_output});
    }

    return {output};
}

std::vector<torch::Tensor> SparseAccumulationOfProducts::backward(
    torch::autograd::AutogradContext *ctx,
    std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto B = saved_variables[1];
    auto C = saved_variables[2];
    auto indices_A = saved_variables[3];
    auto indices_B = saved_variables[4];
    auto indices_output = saved_variables[5];

    auto grad_output = grad_outputs[0];
    if (!grad_output.is_contiguous()) {
        throw std::runtime_error("expected contiguous grad_output");
    }

    auto grad_A = torch::Tensor();
    auto grad_B = torch::Tensor();

    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "sparse_accumulation_of_products_vjp", [&](){
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

            mops::sparse_accumulation_of_products_vjp<scalar_t>(
                mops_grad_A,
                mops_grad_B,
                torch_to_mops_2d<scalar_t>(grad_output),
                torch_to_mops_2d<scalar_t>(A),
                torch_to_mops_2d<scalar_t>(B),
                torch_to_mops_1d<scalar_t>(C),
                torch_to_mops_1d<int32_t>(indices_A),
                torch_to_mops_1d<int32_t>(indices_B),
                torch_to_mops_1d<int32_t>(indices_output)
            );
        });
    } else {
        C10_THROW_ERROR(ValueError,
            "sparse_accumulation_of_products is not implemented for device " + A.device().str()
        );
    }

    return {grad_A, grad_B, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
}
