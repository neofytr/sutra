#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace sutra {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// T dates x N stocks, stored column-major: each stock's history is contiguous,
// so time-series operators stream through memory and vectorise across dates.
struct Panel {
    std::size_t T = 0, N = 0;
    std::vector<float> data;

    Panel(std::size_t t, std::size_t n) : T(t), N(n), data(t * n, kNaN) {}
    float* col(std::size_t j) { return data.data() + j * T; }
    const float* col(std::size_t j) const { return data.data() + j * T; }
    float& operator()(std::size_t t, std::size_t j) { return data[j * T + t]; }
    float operator()(std::size_t t, std::size_t j) const { return data[j * T + t]; }
};

}  // namespace sutra
