#ifndef MOPS_CHECKS_HPP
#define MOPS_CHECKS_HPP

#include "mops/tensor.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

template <typename T_1, size_t N_DIMS_1, typename T_2, size_t N_DIMS_2>
void check_sizes(mops::Tensor<T_1, N_DIMS_1> tensor_1, std::string name_1,
                 size_t dim_1, mops::Tensor<T_2, N_DIMS_2> tensor_2,
                 std::string name_2, size_t dim_2, std::string operation) {
    if (tensor_1.shape[dim_1] != tensor_2.shape[dim_2])
        throw std::runtime_error(
            "Dimension mismatch: the sizes of " + name_1 + " along dimension " +
            std::to_string(dim_1) + " and " + name_2 + " along dimension " +
            std::to_string(dim_2) + " must match in " + operation);
}

template <size_t N_DIMS>
void check_index_tensor(mops::Tensor<int32_t, N_DIMS> tensor, std::string name,
                        size_t max_value, std::string operation) {
    int32_t *data_pointer = tensor.data;
    size_t total_size = 1;
    for (size_t i_dim = 0; i_dim < N_DIMS; i_dim++)
        total_size *= tensor.shape[i_dim];
    int32_t *min_ptr =
        std::min_element(data_pointer, data_pointer + total_size);
    int32_t *max_ptr =
        std::max_element(data_pointer, data_pointer + total_size);
    int32_t min = *min_ptr;
    int32_t max = *max_ptr;
    int32_t max_value_int32 = static_cast<int32_t>(max_value);
    if (max > max_value_int32)
        throw std::runtime_error(
            "Index array " + name + " in operation " + operation +
            " contains elements up to " + std::to_string(max) +
            "; this would cause out-of-bounds accesses. With the " +
            "provided parameters, it can only contain elements up to " +
            std::to_string(max_value));
    if (min < 0)
        throw std::runtime_error("Index array " + name + " in operation " +
                                 operation + " contains negative-valued" +
                                 "indices");
}

#endif
