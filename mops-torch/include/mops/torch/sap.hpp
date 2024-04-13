#ifndef MOPS_TORCH_SAP_H
#define MOPS_TORCH_SAP_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/// TODO
torch::Tensor sparse_accumulation_of_products(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    torch::Tensor indices_A,
    torch::Tensor indices_B,
    torch::Tensor indices_output,
    int64_t output_size
);

class SparseAccumulationOfProducts
    : public torch::autograd::Function<mops_torch::SparseAccumulationOfProducts> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor C,
        torch::Tensor indices_A,
        torch::Tensor indices_B,
        torch::Tensor indices_output,
        int64_t output_size
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

class SparseAccumulationOfProductsBackward
    : public torch::autograd::Function<mops_torch::SparseAccumulationOfProductsBackward> {
  public:
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor grad_output,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor C,
        torch::Tensor indices_A,
        torch::Tensor indices_B,
        torch::Tensor indices_output
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_grad_outputs
    );
};

} // namespace mops_torch

#endif
