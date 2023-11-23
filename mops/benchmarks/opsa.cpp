#include <algorithm>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 20);
    fill_vector_random_floats(B);

    auto indices_output = std::vector<int>(60000);
    fill_vector_random_integers(indices_output, 1000);
    std::sort(indices_output.begin(), indices_output.end());

    auto output = std::vector<double>(1000 * 13 * 20);

    for (int i = 0; i < 1000; i++) {
        mops::outer_product_scatter_add<double>(
            {output.data(), {1000, 13, 20}}, {A.data(), {60000, 13}},
            {B.data(), {60000, 20}}, {indices_output.data(), {60000}});
    }

    return 0;
}
