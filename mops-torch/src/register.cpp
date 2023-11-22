#include <torch/script.h>

#include "mops/torch/opsa.hpp"

TORCH_LIBRARY(mops, m) {
    m.def(
        "outer_product_scatter_add("
            "Tensor A, Tensor B, Tensor indices_output, int output_size"
        ") -> Tensor",
        mops_torch::outer_product_scatter_add
    );
}
