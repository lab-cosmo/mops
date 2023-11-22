#ifndef MOPS_TORCH_OPSA_H
#define MOPS_TORCH_OPSA_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/// TODO
torch::Tensor outer_product_scatter_add(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor indices_output,
    int64_t output_size
);

class OuterProductScatterAdd: public torch::autograd::Function<mops_torch::OuterProductScatterAdd> {
public:
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor indices_output,
        int64_t output_size
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<torch::Tensor> grad_outputs
    );
};

}

#endif
