#include <iostream>
#include <chrono>
#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    size_t size = 1000;
    if (argc > 2) {
        std::cout << "This program only takes one command-line argument\n";
        return 1;
    } else if (argc == 2) {
        size = std::stoi(argv[1]);
    } else {
        // Use default value of 1000
    }
    size *= 32;

    auto A = std::vector<double>(size * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(size * 7);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto indices_A = std::vector<int>(900);
    fill_vector_random_integers(indices_A, 13);

    auto indices_B = std::vector<int>(900);
    fill_vector_random_integers(indices_B, 7);

    auto indices_output = std::vector<int>(900);
    fill_vector_random_integers(indices_output, 100);

    auto O = std::vector<double>(size * 100);

    std::vector<double> execution_times(100);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_of_products<double>(
            {O.data(), {size, 100}}, {A.data(), {size, 13}},
            {B.data(), {size, 7}}, {C.data(), {900}},
            {indices_A.data(), {900}}, {indices_B.data(), {900}},
            {indices_output.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";

    auto grad_output = std::vector<double>(size * 100);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(size * 13);
    auto grad_B = std::vector<double>(size * 7);

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_of_products_vjp<double>(
            {grad_A.data(), {size, 13}}, {grad_B.data(), {size, 7}},
            {grad_output.data(), {size, 100}}, {A.data(), {size, 13}},
            {B.data(), {size, 7}}, {C.data(), {900}},
            {indices_A.data(), {900}}, {indices_B.data(), {900}},
            {indices_output.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
