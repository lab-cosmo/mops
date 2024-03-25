#ifndef MOPS_CHECKS_HPP
#define MOPS_CHECKS_HPP

#include "mops/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

/// Check that the sizes of two tensors (along a given dimension for each)
/// are equal. The template parameters T_1 and T_2 are the types of the
/// elements of the two tensors, and N_DIMS_1 and N_DIMS_2 are the number of
/// dimensions of the two tensors, respectively.
/// ``tensor_1`` and ``tensor_2`` are the two tensors being compared;
/// ``dim_1`` and ``dim_2`` are the dimensions along which the sizes are
/// being compared. Additionally, names are provided for the two tensors
/// and the operation being performed to make the error message more informative.
/// Note that T_1 and T_2 can be different types, such as when comparing
/// an index tensor (int32_t) with a data tensor (float/double).
template <typename T_1, size_t N_DIMS_1, typename T_2, size_t N_DIMS_2>
void check_sizes(
    mops::Tensor<T_1, N_DIMS_1> tensor_1,
    const std::string& name_1,
    size_t dim_1,
    mops::Tensor<T_2, N_DIMS_2> tensor_2,
    const std::string& name_2,
    size_t dim_2,
    const std::string& operation
) {
    if (tensor_1.shape[dim_1] != tensor_2.shape[dim_2]) {
        throw std::runtime_error(
            "Dimension mismatch: the sizes of " + name_1 + " along dimension " +
            std::to_string(dim_1) + " and " + name_2 + " along dimension " + std::to_string(dim_2) +
            " must match in " + operation
        );
    }
}

/// Checks that an index tensor does not contain out-of-bounds indices.
/// The template parameter N_DIMS is the number of dimensions of the tensor.
/// ``tensor`` is the tensor being checked, ``name`` is the name of the tensor,
/// ``max_value`` is the maximum value that the indices can take, and
/// ``operation`` is the operation being performed. If the tensor contains
/// out-of-bounds indices, a runtime error is thrown.
template <size_t N_DIMS>
void check_index_tensor(
    mops::Tensor<int32_t, N_DIMS> tensor,
    const std::string& name,
    size_t max_value,
    const std::string& operation
) {
    int32_t *data_pointer = tensor.data;
    size_t total_size = 1;
    for (size_t i_dim = 0; i_dim < N_DIMS; i_dim++) {
        total_size *= tensor.shape[i_dim];
    }
    int32_t *min_ptr = std::min_element(data_pointer, data_pointer + total_size);
    int32_t *max_ptr = std::max_element(data_pointer, data_pointer + total_size);
    int32_t min = *min_ptr;
    int32_t max = *max_ptr;
    int32_t max_value_int32 = static_cast<int32_t>(max_value);
    if (max >= max_value_int32) {
        throw std::runtime_error(
            "Index array " + name + " in operation " + operation + " contains elements up to " +
            std::to_string(max) + "; this would cause out-of-bounds accesses. With the " +
            "provided parameters, it can only contain elements up to " +
            std::to_string(max_value - 1)
        );
    }
    if (min < 0) {
        throw std::runtime_error(
            "Index array " + name + " in operation " + operation + " contains negative-valued" +
            "indices"
        );
    }
}

/// Checks that the shapes of two tensors are the same. The template parameter
/// T is the type of the elements of the tensors, and N_DIMS is the number of
/// dimensions of the tensors. ``tensor_1`` and ``tensor_2`` are the two tensors
/// being compared, and ``name_1`` and ``name_2`` are the names of the two
/// tensors. Additionally, the name of the operation being performed is provided
/// to make the error message more informative.
template <typename T, size_t N_DIMS>
void check_same_shape(
    mops::Tensor<T, N_DIMS> tensor_1,
    const std::string& name_1,
    mops::Tensor<T, N_DIMS> tensor_2,
    const std::string& name_2,
    const std::string& operation
) {
    for (size_t i_dim = 0; i_dim < N_DIMS; i_dim++) {
        if (tensor_1.shape[i_dim] != tensor_2.shape[i_dim]) {
            throw std::runtime_error(
                "Dimension mismatch: the sizes of " + name_1 + " and " + name_2 + " must match in " +
                operation + ", but they are " + std::to_string(tensor_1.shape[i_dim]) + " and " +
                std::to_string(tensor_2.shape[i_dim]) + " respectively" + " along dimension " +
                std::to_string(i_dim)
            );
        }
    }
}

#endif
