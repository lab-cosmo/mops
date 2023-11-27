#include <cmath>
#include <numeric>
#include <random>

void fill_vector_random_integers(std::vector<int> &vector, size_t n) {
    // Random engine seeded with current time
    std::mt19937 engine(time(nullptr));

    // Uniform distribution from 0 to N-1
    std::uniform_int_distribution<int> distribution(0, n - 1);

    // Fill the vector with random numbers
    for (auto &num : vector) {
        num = distribution(engine);
    }
}

template <typename scalar_t>
void fill_vector_random_floats(std::vector<scalar_t> &vector) {
    // Random engine seeded with current time
    std::mt19937 engine(time(nullptr));

    // Uniform distribution from -1 to 1
    std::uniform_real_distribution<scalar_t> distribution(-1.0, 1.0);

    // Fill the vector with random numbers
    for (auto &num : vector) {
        num = distribution(engine);
    }
}

std::pair<double, double>
calculate_mean_and_stddev(const std::vector<double> &times) {
    double mean =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    double sum_sq_diff = 0.0;
    for (const auto &time : times) {
        sum_sq_diff += (time - mean) * (time - mean);
    }
    double stddev = std::sqrt(sum_sq_diff / times.size());

    return {mean, stddev};
}
