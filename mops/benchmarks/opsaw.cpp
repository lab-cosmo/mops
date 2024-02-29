#include <algorithm>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 20);
    fill_vector_random_floats(B);

    auto W = std::vector<double>(1000 * 20);
    fill_vector_random_floats(W);

    auto indices_W = std::vector<int32_t>(60000);
    fill_vector_random_integers(indices_W, 1000);

    auto indices_output = std::vector<int32_t>(60000);
    fill_vector_random_integers(indices_output, 1000);

    auto O = std::vector<double>(1000 * 13 * 20);

    for (int i = 0; i < 1000; i++) {
        mops::outer_product_scatter_add_with_weights<double>(
            {O.data(), {1000, 13, 20}},
            {A.data(), {60000, 13}},
            {B.data(), {60000, 20}},
            {W.data(), {1000, 20}},
            {indices_W.data(), {60000}},
            {indices_output.data(), {60000}}
        );
    }

    return 0;
}
