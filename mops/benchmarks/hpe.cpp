#include <vector>

#include "mops.hpp"
#include "utils.hpp"

int main() {

    auto A = std::vector<double>(1000 * 2000);
    fill_vector_random_floats(A);

    auto C = std::vector<double>(100000);
    fill_vector_random_floats(C);

    auto P = std::vector<int>(100000 * 4);
    fill_vector_random_integers(P, 2000);

    auto O = std::vector<double>(1000);

    for (int i = 0; i < 1000; i++) {
        mops::homogeneous_polynomial_evaluation<double>(
            {O.data(), {1000}}, {A.data(), {1000, 2000}}, {C.data(), {100000}},
            {P.data(), {100000, 4}}
        );
    }

    return 0;
}
