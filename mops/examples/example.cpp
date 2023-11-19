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
    auto P = std::vector<int>(100 * 4);
    auto O = std::vector<double>(100);

    mops::homogeneous_polynomial_evaluation<double>(
        {O.data(), {100}}, {A.data(), {100, 100}}, {C.data(), {100}},
        {P.data(), {100, 4}});

    // Another example, with a different function:
    A = std::vector<double>(100 * 10);
    auto B = std::vector<double>(100 * 20);
    P = std::vector<int>(100);
    O = std::vector<double>(10 * 10 * 20);

    mops::outer_product_scatter_add<double>(
        {O.data(), {10, 10, 20}}, {A.data(), {100, 10}}, {B.data(), {100, 20}},
        {P.data(), {100}});

    return 0;
}
