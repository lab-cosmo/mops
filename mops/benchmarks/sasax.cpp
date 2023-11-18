#include <algorithm>
#include <iostream>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto O = std::vector<double>(1000 * 100 * 32);
    fill_vector_random_floats(O);

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto R = std::vector<double>(60000 * 32);
    fill_vector_random_floats(R);

    auto X = std::vector<double>(1000 * 7 * 32);
    fill_vector_random_floats(A);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto I = std::vector<int>(60000);
    fill_vector_random_integers(I, 1000);
    std::sort(I.begin(), I.end());

    auto J = std::vector<int>(60000);
    fill_vector_random_integers(J, 1000);

    auto M_1 = std::vector<int>(900);
    fill_vector_random_integers(M_1, 13);

    auto M_2 = std::vector<int>(900);
    fill_vector_random_integers(M_2, 7);

    auto M_3 = std::vector<int>(900);
    fill_vector_random_integers(M_3, 100);

    for (int i = 0; i < 10; i++) {
        mops::sparse_accumulation_scatter_add_with_weights<double>(
            {O.data(), {1000 * 100 * 32}}, {A.data(), {60000, 13}},
            {R.data(), {60000, 32}}, {X.data(), {1000, 7, 32}},
            {C.data(), {900}}, {I.data(), {60000}}, {J.data(), {60000}},
            {M_1.data(), {900}}, {M_2.data(), {900}}, {M_3.data(), {900}});
    }

    return 0;
}
