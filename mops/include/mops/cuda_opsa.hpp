#ifndef MOPS_OPSA_CUDA
#define MOPS_OPSA_CUDA

#include <cstddef>
#include <cstdint>

#include "mops/exports.h"
#include "mops/tensor.hpp"

namespace mops {

namespace cuda {

template <typename scalar_t>
void outer_product_scatter_add_cuda(
    const scalar_t *A,             // [nedges, nfeatures_A]
    const scalar_t *B,             // [nedges, nfeatures_B]
    const int32_t nnodes,          // number of nodes we're summing into
    const int32_t nedges,          // number of edges -> batch size of A and B
    const int32_t nfeatures_A,     // number of features of A
    const int32_t nfeatures_B,     // number of features of B
    const int32_t *indices_output, // sorted list of indices to sum
                                   // into [nedges]
    scalar_t *__restrict__ output  // shape: [nnodes, nfeatures_B, nfeatures_A]
    // -> this ordering because contiguity of threadCol

);

template <typename scalar_t>
void outer_product_scatter_add_vjp_cuda(
    const scalar_t *A,             // [nedges, nfeatures_A]
    const scalar_t *B,             // [nedges, nfeatures_B]
    const int32_t nnodes,          // number of nodes we're summing into
    const int32_t nedges,          // number of edges -> batch size of A and B
    const int32_t nfeatures_A,     // number of features of A
    const int32_t nfeatures_B,     // number of features of B
    const int32_t *indices_output, // sorted list of indices to
                                   // sum into [nedges]
    scalar_t *grad_in,             // grad_input: [nnodes, nfeatures_B,
                                   // nfeatures_A]
    scalar_t *grad_A,              // [nedges, nfeatures_A],
    scalar_t *grad_B               // [nedges, nfeatures_B]
);

} // namespace cuda
} // namespace mops

#endif
