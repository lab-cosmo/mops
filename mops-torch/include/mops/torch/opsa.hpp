#ifndef MOPS_TORCH_OPSA_H
#define MOPS_TORCH_OPSA_H

#include <torch/script.h>

#include <mops.hpp>

namespace mops_torch {

/*
 * Outer-Product-Scatter-Add (OPSA)
 * Computes the outer product between tensors A, B along the last dimension, and sums the result
 * into a new tensor of shape [output_size, A.shape[1], B.shape[1]], where the summation index
 * is given by the tensor indices_output.
 *
 * For example, If A has shape (5, 32) and B has shape (5, 16), and indices_output contains
 * [0, 0, 1, 1, 2], the output will have shape (3, 32, 16). For example using numpy terminology, the
 * value of output[0] in this case would be equal to
 * output[0, :, :] = A[0, :, None] * B[0, None, :] + A[1, :, None] * B[1, None,  :]
 */
torch::Tensor outer_product_scatter_add(
    torch::Tensor A, torch::Tensor B, torch::Tensor indices_output, int64_t output_size
);

class OuterProductScatterAdd : public torch::autograd::Function<mops_torch::OuterProductScatterAdd> {
  public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor indices_output,
        int64_t output_size
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

class OuterProductScatterAddBackward
    : public torch::autograd::Function<mops_torch::OuterProductScatterAddBackward> {
  public:
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor grad_output,
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor indices_output
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx, std::vector<torch::Tensor> grad_outputs
    );
};

} // namespace mops_torch

#endif
