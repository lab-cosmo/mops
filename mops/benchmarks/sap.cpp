#include "mops.hpp"
#include <vector>
#include <chrono>
#include <random>
#include <ctime>


void fill_vector_random_integers(std::vector<int>& vector, size_t n) {
    // Random engine seeded with current time
    std::mt19937 engine(time(nullptr));

    // Uniform distribution from 0 to N-1
    std::uniform_int_distribution<int> distribution(0, n-1);

    // Fill the vector with random numbers
    for (auto& num : vector) {
        num = distribution(engine);
    }
}

template<typename scalar_t>
void fill_vector_random_floats(std::vector<scalar_t>& vector) {
    // Random engine seeded with current time
    std::mt19937 engine(time(nullptr));

    // Uniform distribution from -1 to 1
    std::uniform_real_distribution<scalar_t> distribution(-1.0, 1.0);

    // Fill the vector with random numbers
    for (auto& num : vector) {
        num = distribution(engine);
    }
}

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

    for (int i=0; i<10000; i++) mops::sparse_accumulation_of_products<double>(
        {O.data(), {1000, 50}}, {A.data(), {1000, 20}}, {B.data(), {1000, 6}},
        {C.data(), {100}}, {P_A.data(), {100}}, {P_B.data(), {100}}, {P_O.data(), {100}});

    return 0;
}