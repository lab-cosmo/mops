#include "mops/utils.hpp"
#include <algorithm>
#include <execution>
#include <vector>

std::vector<int32_t> find_first_occurrences(const int32_t *scatter_indices,
                                            size_t scatter_size,
                                            size_t output_dim) {
    // Finds the positions of the first occurrences within scatter_indices (size
    // scatter_size) Out_dim is the highest index we are looking for in
    // scatter_indices. It must be greater or equal to its max

    std::vector<int32_t> first_occurrences =
        std::vector<int32_t>(output_dim + 1, -1);
    first_occurrences[scatter_indices[0]] = 0;

    std::vector<size_t> indices(scatter_size - 1);
    std::iota(indices.begin(), indices.end(), 0);
    std::for_each(std::execution::par, indices.begin(), indices.end(),
                  [&](size_t i) {
                      if (scatter_indices[i] < scatter_indices[i + 1])
                          first_occurrences[scatter_indices[i + 1]] = i + 1;
                  });
    first_occurrences[output_dim] = scatter_size;
    for (int i = output_dim; i > -1;
         i--) { // size_t always positive, wouldn't stop loop
        if (first_occurrences[i] == -1)
            first_occurrences[i] = first_occurrences[i + 1];
    }

    return first_occurrences;
}
