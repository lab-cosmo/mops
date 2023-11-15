#ifndef MOPS_UTILS_HPP
#define MOPS_UTILS_HPP

#include <cstdint>
#include <cstddef>
#include <vector>

std::vector<int32_t> find_first_occurrences(const int32_t* scatter_indices, size_t scatter_size, size_t output_dim);

#endif
