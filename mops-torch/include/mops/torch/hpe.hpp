#ifndef MOPS_TORCH_HPE_H
#define MOPS_TORCH_HPE_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/// TODO
torch::Tensor homogeneous_polynomial_evaluation(
    torch::Tensor A, torch::Tensor C, torch::Tensor indices_A
);

class HomogeneousPolynomialEvaluation
    : public torch::autograd::Function<mops_torch::HomogeneousPolynomialEvaluation> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx, torch::Tensor A, torch::Tensor C, torch::Tensor indices_A
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

class HomogeneousPolynomialEvaluationBackward
    : public torch::autograd::Function<mops_torch::HomogeneousPolynomialEvaluationBackward> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor grad_output,
        torch::Tensor A,
        torch::Tensor C,
        torch::Tensor indices_A
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

} // namespace mops_torch

#endif
