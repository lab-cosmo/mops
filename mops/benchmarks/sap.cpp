#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(1000 * 20);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(1000 * 6);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(100);
    fill_vector_random_floats(C);

    auto P_A = std::vector<int>(100);
    fill_vector_random_integers(P_A, 20);

    auto P_B = std::vector<int>(100);
    fill_vector_random_integers(P_B, 6);

    auto P_O = std::vector<int>(100);
    fill_vector_random_integers(P_O, 50);

    auto O = std::vector<double>(1000 * 50);

    for (int i = 0; i < 10000; i++)
        mops::sparse_accumulation_of_products<double>(
            {O.data(), {1000, 50}}, {A.data(), {1000, 20}},
            {B.data(), {1000, 6}}, {C.data(), {100}}, {P_A.data(), {100}},
            {P_B.data(), {100}}, {P_O.data(), {100}});

    return 0;
}
