#include <algorithm>
#include <iostream>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto output = std::vector<double>(1000 * 100 * 32);
    fill_vector_random_floats(output);

    auto A = std::vector<double>(60000 * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(60000 * 32);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto W = std::vector<double>(1000 * 7 * 32);
    fill_vector_random_floats(A);

    auto indices_A = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_A, 13);

    auto indices_W_1 = std::vector<int32_t>(60000);
    fill_vector_random_integers(indices_W_1, 1000);

    auto indices_W_2 = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_W_2, 7);

    auto indices_output_1 = std::vector<int32_t>(60000);
    fill_vector_random_integers(indices_output_1, 1000);

    auto indices_output_2 = std::vector<int32_t>(900);
    fill_vector_random_integers(indices_output_2, 100);

    for (int i = 0; i < 100; i++) {
        mops::sparse_accumulation_scatter_add_with_weights<double>(
            {output.data(), {1000, 100, 32}},
            {A.data(), {60000, 13}},
            {B.data(), {60000, 32}},
            {C.data(), {900}},
            {W.data(), {1000, 7, 32}},
            {indices_A.data(), {900}},
            {indices_W_1.data(), {60000}},
            {indices_W_2.data(), {900}},
            {indices_output_1.data(), {60000}},
            {indices_output_2.data(), {900}}
        );
    }

    return 0;
}
