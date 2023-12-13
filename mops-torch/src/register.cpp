#include <torch/script.h>

#include "mops/torch.hpp"

TORCH_LIBRARY(mops, m) {
    m.def(
        "homogeneous_polynomial_evaluation("
        "Tensor A, Tensor C, Tensor indices_A"
        ") -> Tensor",
        mops_torch::homogeneous_polynomial_evaluation
    );
    m.def(
        "outer_product_scatter_add("
        "Tensor A, Tensor B, Tensor indices_output, int output_size"
        ") -> Tensor",
        mops_torch::outer_product_scatter_add
    );
    m.def(
        "sparse_accumulation_of_products("
        "Tensor A, Tensor B, Tensor C, Tensor indices_A, Tensor indices_B, "
        "Tensor indices_output, int output_size"
        ") -> Tensor",
        mops_torch::sparse_accumulation_of_products
    );
    m.def(
        "outer_product_scatter_add_with_weights("
        "Tensor A, Tensor B, Tensor W, Tensor indices_W, Tensor indices_output"
        ") -> Tensor",
        mops_torch::outer_product_scatter_add_with_weights
    );
    m.def(
        "sparse_accumulation_scatter_add_with_weights("
        "Tensor A, Tensor B, Tensor C, Tensor W, Tensor indices_A, Tensor "
        "indices_W_1, Tensor indices_W_2, Tensor indices_output_1, Tensor "
        "indices_output_2, int output_size"
        ") -> Tensor",
        mops_torch::sparse_accumulation_scatter_add_with_weights
    );
}
