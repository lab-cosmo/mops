#ifndef MOPS_TORCH_SASAW_H
#define MOPS_TORCH_SASAW_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/// TODO
torch::Tensor sparse_accumulation_scatter_add_with_weights(
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
);

class SparseAccumulationScatterAddWithWeights
    : public torch::autograd::Function<mops_torch::SparseAccumulationScatterAddWithWeights> {
  public:
    static torch::Tensor forward(
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
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> grad_outputs
    );
};

} // namespace mops_torch

#endif
