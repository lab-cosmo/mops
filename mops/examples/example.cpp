#include "mops.hpp"
#include <vector>

int main() {
    // To avoid calls with a very large number of arguments,
    // mops uses a mops::Tensor<T, N_DIMS> struct which simply
    // consists a data pointer and a shape in the form of a std::array.
    //
    // All mops operations take mops::Tensor objects as their
    // inputs, and these can be initialized implicitly during
    // the call in the following way:

    auto A = std::vector<double>(100 * 100);
    auto C = std::vector<double>(100);
    auto indices_A = std::vector<int>(100 * 4);
    auto output = std::vector<double>(100);

    mops::homogeneous_polynomial_evaluation<double>(
        {output.data(), {100}}, {A.data(), {100, 100}}, {C.data(), {100}}, {indices_A.data(), {100, 4}}
    );

    return 0;
}
