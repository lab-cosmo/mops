#include <algorithm>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto R = std::vector<double>(60000 * 20);
    fill_vector_random_floats(R);

    auto X = std::vector<double>(1000 * 20);
    fill_vector_random_floats(X);

    auto I = std::vector<int>(60000);
    fill_vector_random_integers(I, 1000);
    std::sort(I.begin(), I.end());

    auto J = std::vector<int>(60000);
    fill_vector_random_integers(J, 1000);

    auto O = std::vector<double>(1000 * 13 * 20);

    for (int i = 0; i < 1000; i++) {
        mops::outer_product_scatter_add_with_weights<double>(
            {O.data(), {1000, 13, 20}}, {A.data(), {60000, 13}},
            {R.data(), {60000, 20}}, {X.data(), {60000, 20}},
            {I.data(), {60000}}, {J.data(), {60000}});
    }

    return 0;
}
