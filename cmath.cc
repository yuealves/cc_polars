#include <pybind11/pybind11.h>
#include <cmath>

long long gcd(long long a, long long b) {
    if (b == 0) {
        return a;
    }
    return gcd(b, a % b);
}

long long lcm(long long a, long long b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return std::abs(a * b) / gcd(a, b);
}

namespace py = pybind11;

PYBIND11_MODULE(my_module, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("gcd", &gcd, "A function that calculates the greatest common divisor of two numbers.");
    m.def("lcm", &lcm, "A function that calculates the least common multiple of two numbers.");
}
