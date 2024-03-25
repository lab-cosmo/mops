#ifndef MOPS_TORCH_OPSAW_H
#define MOPS_TORCH_OPSAW_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/// TODO
torch::Tensor outer_product_scatter_add_with_weights(
    torch::Tensor A, torch::Tensor B, torch::Tensor W, torch::Tensor indices_W, torch::Tensor indices_output
);

class OuterProductScatterAddWithWeights
    : public torch::autograd::Function<mops_torch::OuterProductScatterAddWithWeights> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor W,
        torch::Tensor indices_W,
        torch::Tensor indices_output
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

} // namespace mops_torch

#endif
