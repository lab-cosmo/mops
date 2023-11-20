#include <algorithm>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(32000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(32000 * 7);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto P_A = std::vector<int>(900);
    fill_vector_random_integers(P_A, 13);

    auto P_B = std::vector<int>(900);
    fill_vector_random_integers(P_B, 7);

    auto P_O = std::vector<int>(900);
    fill_vector_random_integers(P_O, 100);
    std::sort(P_O.begin(), P_O.end());

    auto O = std::vector<double>(32000 * 100);

    for (int i = 0; i < 1000; i++)
        mops::sparse_accumulation_of_products<double>(
            {O.data(), {32000, 100}}, {A.data(), {32000, 13}},
            {B.data(), {32000, 7}}, {C.data(), {900}}, {P_A.data(), {900}},
            {P_B.data(), {900}}, {P_O.data(), {900}});

    return 0;
}
