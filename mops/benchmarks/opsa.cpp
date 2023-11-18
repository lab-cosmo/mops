#include <algorithm>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 20);
    fill_vector_random_floats(B);

    auto P = std::vector<int>(60000);
    fill_vector_random_integers(P, 1000);
    std::sort(P.begin(), P.end());

    auto O = std::vector<double>(1000 * 13 * 20);

    for (int i = 0; i < 1000; i++) {
        mops::outer_product_scatter_add<double>(
            {O.data(), {1000, 13, 20}}, {A.data(), {60000, 13}},
            {B.data(), {60000, 20}}, {P.data(), {60000}});
    }

    return 0;
}
