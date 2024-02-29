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

    auto indices_A = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_A, 13);

    auto indices_B = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_B, 7);

    auto indices_output = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_output, 100);

    auto O = std::vector<double>(32000 * 100);

    for (int i = 0; i < 1000; i++) {
        mops::sparse_accumulation_of_products<double>(
            {O.data(), {32000, 100}},
            {A.data(), {32000, 13}},
            {B.data(), {32000, 7}},
            {C.data(), {900}},
            {indices_A.data(), {900}},
            {indices_B.data(), {900}},
            {indices_output.data(), {900}}
        );
    }

    return 0;
}
