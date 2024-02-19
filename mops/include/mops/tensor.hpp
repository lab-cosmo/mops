#ifndef MOPS_TENSOR_HPP
#define MOPS_TENSOR_HPP

#include <cstddef>

namespace mops {
template <typename scalar_t, size_t N_DIMS> struct Tensor {
    /// Pointer to the first element of the tensor. The data must be
    /// contiguous and in row-major order.
    scalar_t *MOPS_RESTRICT data;
    /// Shape of the tensor
    size_t shape[N_DIMS];
};
} // namespace mops

#endif
