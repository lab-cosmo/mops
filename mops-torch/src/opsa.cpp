#include "mops/torch/opsa.hpp"

using namespace mops_torch;

torch::Tensor
mops_torch::outer_product_scatter_add(torch::Tensor A, torch::Tensor B,
                                      torch::Tensor indices_output,
                                      int64_t output_size) {
    return OuterProductScatterAdd::apply(A, B, indices_output, output_size)[0];
}

template <typename scalar_t>
static mops::Tensor<scalar_t, 1> torch_to_mops_1d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 1);
    return {
        tensor.data_ptr<scalar_t>(),
        {static_cast<size_t>(tensor.size(0))},
    };
}

template <typename scalar_t>
static mops::Tensor<scalar_t, 2> torch_to_mops_2d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 2);
    return {
        tensor.data_ptr<scalar_t>(),
        {static_cast<size_t>(tensor.size(0)),
         static_cast<size_t>(tensor.size(1))},
    };
}

template <typename scalar_t>
static mops::Tensor<scalar_t, 3> torch_to_mops_3d(torch::Tensor tensor) {
    assert(tensor.sizes().size() == 3);
    return {
        tensor.data_ptr<scalar_t>(),
        {static_cast<size_t>(tensor.size(0)),
         static_cast<size_t>(tensor.size(1)),
         static_cast<size_t>(tensor.size(2))},
    };
}

std::vector<torch::Tensor> OuterProductScatterAdd::forward(
    torch::autograd::AutogradContext *ctx, torch::Tensor A, torch::Tensor B,
    torch::Tensor indices_output, int64_t output_size) {
    if (A.sizes().size() != 2 || B.sizes().size() != 2) {
        C10_THROW_ERROR(ValueError, "`A` and `B` must be 2-D tensor");
    }

    if (indices_output.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError, "`indices_output` must be a 1-D tensor");
    }

    if (indices_output.scalar_type() != torch::kInt32) {
        C10_THROW_ERROR(ValueError,
                        "`indices_output` must be a tensor of 32-bit integers");
    }

    if (A.device() != B.device() || A.device() != indices_output.device()) {
        C10_THROW_ERROR(ValueError,
                        "all tensors must be on the same device, got " +
                            A.device().str() + ", " + B.device().str() +
                            ", and " + indices_output.device().str());
    }

    if (A.scalar_type() != B.scalar_type()) {
        C10_THROW_ERROR(
            ValueError,
            std::string("`A` and `B` must have the same dtype, got ") +
                torch::toString(A.scalar_type()) + " and " +
                torch::toString(B.scalar_type()));
    }

    torch::Tensor output;
    if (A.device().is_cpu()) {
        output = torch::zeros(
            {output_size, A.size(1), B.size(1)},
            torch::TensorOptions().dtype(A.scalar_type()).device(A.device()));

        assert(output.is_contiguous());

        AT_DISPATCH_FLOATING_TYPES(
            A.scalar_type(), "outer_product_scatter_add", [&]() {
                mops::outer_product_scatter_add<scalar_t>(
                    torch_to_mops_3d<scalar_t>(output),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "outer_product_scatter_add is not implemented for device " +
                A.device().str());
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
                auto mops_grad_A = mops::Tensor<scalar_t, 3>{nullptr, {0, 0, 0}};
                if (A.requires_grad()) {
                    grad_A = torch::zeros_like(A);
                    mops_grad_A = torch_to_mops_3d<scalar_t>(grad_A);
                }

                auto mops_grad_B = mops::Tensor<scalar_t, 3>{nullptr, {0, 0, 0}};
                if (B.requires_grad()) {
                    grad_B = torch::zeros_like(B);
                    mops_grad_B = torch_to_mops_3d<scalar_t>(grad_B);
                }

                mops::outer_product_scatter_add_vjp<scalar_t>(
                    mops_grad_A, mops_grad_B,
                    torch_to_mops_3d<scalar_t>(grad_output),
                    torch_to_mops_2d<scalar_t>(A),
                    torch_to_mops_2d<scalar_t>(B),
                    torch_to_mops_1d<int32_t>(indices_output));
            });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "outer_product_scatter_add is not implemented for device " +
                A.device().str());
    }

    return {grad_A, grad_B, torch::Tensor(), torch::Tensor()};
}
