#include "mops/torch/hpe.hpp"
#include "mops/torch/utils.hpp"

using namespace mops_torch;

torch::Tensor mops_torch::homogeneous_polynomial_evaluation(
    torch::Tensor A, torch::Tensor C, torch::Tensor indices_A
) {
    return HomogeneousPolynomialEvaluation::apply(A, C, indices_A);
}

torch::Tensor HomogeneousPolynomialEvaluation::forward(
    torch::autograd::AutogradContext* ctx, torch::Tensor A, torch::Tensor C, torch::Tensor indices_A
) {
    details::check_all_same_device({A, C, indices_A});
    details::check_floating_dtype({A, C});
    details::check_integer_dtype({indices_A});
    details::check_number_of_dimensions(A, 2, "A", "homogeneous_polynomial_evaluation");
    details::check_number_of_dimensions(C, 1, "C", "homogeneous_polynomial_evaluation");
    details::check_number_of_dimensions(
        indices_A, 2, "indices_A", "homogeneous_polynomial_evaluation"
    );
    // Shape consistency checks are performed inside
    // mops::homogeneous_polynomial_evaluation

    torch::Tensor output =
        torch::empty({A.size(0)}, torch::TensorOptions().dtype(A.scalar_type()).device(A.device()));

    if (A.device().is_cpu()) {

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "homogeneous_polynomial_evaluation", [&]() {
            mops::homogeneous_polynomial_evaluation<scalar_t>(
                details::torch_to_mops_1d<scalar_t>(output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_2d<int32_t>(indices_A)
            );
        });
    }
    if (A.device().is_cuda()) {

        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "homogeneous_polynomial_evaluation", [&]() {
            mops::cuda::homogeneous_polynomial_evaluation<scalar_t>(
                details::torch_to_mops_1d<scalar_t>(output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_2d<int32_t>(indices_A)
            );
        });

    } else {
        C10_THROW_ERROR(
            ValueError,
            "homogeneous_polynomial_evaluation is not implemented for device " + A.device().str()
        );
    }

    if (A.requires_grad()) {
        ctx->save_for_backward({A, C, indices_A});
    }

    return {output};
}

std::vector<torch::Tensor> HomogeneousPolynomialEvaluation::backward(
    torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
) {
    auto saved_variables = ctx->get_saved_variables();
    auto A = saved_variables[0];
    auto C = saved_variables[1];
    auto indices_A = saved_variables[2];

    auto grad_output = grad_outputs[0].contiguous();

    if (C.requires_grad()) {
        C10_THROW_ERROR(
            ValueError,
            "gradients with respect to C are not supported "
            "in homogeneous_polynomial_evaluation"
        );
    }

    auto grad_A = torch::Tensor();
    if (A.device().is_cpu()) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "homogeneous_polynomial_evaluation_vjp", [&]() {
            auto mops_grad_A = mops::Tensor<scalar_t, 2>{nullptr, {0, 0}};
            if (A.requires_grad()) {
                grad_A = torch::empty_like(A);
                mops_grad_A = details::torch_to_mops_2d<scalar_t>(grad_A);
            }

            mops::homogeneous_polynomial_evaluation_vjp<scalar_t>(
                mops_grad_A,
                details::torch_to_mops_1d<scalar_t>(grad_output),
                details::torch_to_mops_2d<scalar_t>(A),
                details::torch_to_mops_1d<scalar_t>(C),
                details::torch_to_mops_2d<int32_t>(indices_A)
            );
        });
    } else {
        C10_THROW_ERROR(
            ValueError,
            "homogeneous_polynomial_evaluation is not implemented for device " + A.device().str()
        );
    }

    return {grad_A, torch::Tensor(), torch::Tensor()};
}
