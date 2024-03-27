#include <vector>

#include "mops/tensor.hpp"

#include "utils.hpp"

std::vector<std::vector<size_t>> get_write_list(mops::Tensor<int32_t, 1> write_coordinates) {
    std::vector<std::vector<size_t>> write_list(write_coordinates.shape[0]);
    int32_t* write_coordinates_data = write_coordinates.data;
    for (size_t i = 0; i < write_coordinates.shape[0]; i++) {
        write_list[write_coordinates_data[i]].push_back(i);
    }
    return write_list;
}
