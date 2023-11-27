#include <chrono>
#include <iostream>
#include <vector>
#include <tbb/task_scheduler_init.h>

#include "mops.hpp"
#include "utils.hpp"

int main(int argc, char **argv) {
    const char* omp_num_threads_str = std::getenv("OMP_NUM_THREADS");
    int number_of_threads = omp_num_threads_str ? std::stoi(omp_num_threads_str) : tbb::task_scheduler_init::automatic;
    tbb::task_scheduler_init init(number_of_threads);

    size_t output_size = 1000;
    if (argc > 2) {
        std::cout << "This program only takes one command-line argument\n";
        return 1;
    } else if (argc == 2) {
        output_size = std::stoi(argv[1]);
    } else {
        // Use default value of 1000
    }
    size_t input_size = 60 * output_size;

    auto output = std::vector<double>(output_size * 100 * 32);
    fill_vector_random_floats(output);

    auto A = std::vector<double>(input_size * 13);
    fill_vector_random_floats(A);

    auto B = std::vector<double>(input_size * 32);
    fill_vector_random_floats(B);

    auto C = std::vector<double>(900);
    fill_vector_random_floats(C);

    auto W = std::vector<double>(output_size * 7 * 32);
    fill_vector_random_floats(A);

    auto indices_A = std::vector<int>(900);
    fill_vector_random_integers(indices_A, 13);

    auto indices_W_1 = std::vector<int>(input_size);
    fill_vector_random_integers(indices_W_1, output_size);

    auto indices_W_2 = std::vector<int>(900);
    fill_vector_random_integers(indices_W_2, 7);

    auto indices_output_1 = std::vector<int>(input_size);
    fill_vector_random_integers(indices_output_1, output_size);

    auto indices_output_2 = std::vector<int>(900);
    fill_vector_random_integers(indices_output_2, 100);

    auto execution_times = std::vector<double>(100);
    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_scatter_add_with_weights<double>(
            {output.data(), {output_size, 100, 32}},
            {A.data(), {input_size, 13}}, {B.data(), {input_size, 32}},
            {C.data(), {900}}, {W.data(), {output_size, 7, 32}},
            {indices_A.data(), {900}}, {indices_W_1.data(), {input_size}},
            {indices_W_2.data(), {900}},
            {indices_output_1.data(), {input_size}},
            {indices_output_2.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean, stddev] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time: " << mean << " ms\n";
    std::cout << "Standard Deviation: " << stddev << " ms\n";

    auto grad_output = std::vector<double>(output_size * 100 * 32);
    fill_vector_random_floats(grad_output);

    auto grad_A = std::vector<double>(input_size * 13);
    auto grad_B = std::vector<double>(input_size * 32);
    auto grad_W = std::vector<double>(output_size * 7 * 32);

    for (int i = 0; i < 100; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        mops::sparse_accumulation_scatter_add_with_weights_vjp<double>(
            {grad_A.data(), {input_size, 13}},
            {grad_B.data(), {input_size, 32}},
            {grad_W.data(), {output_size, 7, 32}},
            {grad_output.data(), {output_size, 100, 32}},
            {A.data(), {input_size, 13}}, {B.data(), {input_size, 32}},
            {C.data(), {900}}, {W.data(), {output_size, 7, 32}},
            {indices_A.data(), {900}}, {indices_W_1.data(), {input_size}},
            {indices_W_2.data(), {900}},
            {indices_output_1.data(), {input_size}},
            {indices_output_2.data(), {900}});
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        execution_times[i] = elapsed.count();
    }

    auto [mean_vjp, stddev_vjp] = calculate_mean_and_stddev(execution_times);

    std::cout << "Average Time vjp: " << mean_vjp << " ms\n";
    std::cout << "Standard Deviation vjp: " << stddev_vjp << " ms\n";

    return 0;
}
