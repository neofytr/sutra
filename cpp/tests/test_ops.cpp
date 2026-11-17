#include <cassert>
#include <cmath>
#include <cstdio>

#include "sutra/ops.hpp"

using namespace sutra;

static bool close(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    Panel x(6, 2);
    for (std::size_t t = 0; t < 6; ++t) { x(t, 0) = float(t + 1); x(t, 1) = float(10 - t); }
    x(3, 1) = kNaN;

    Panel m = ts_mean(x, 3);
    assert(std::isnan(m(1, 0)) && close(m(2, 0), 2.f) && close(m(5, 0), 5.f));
    assert(std::isnan(m(3, 1)) && std::isnan(m(5, 1)));  // NaN at t=3 blocks windows through t=5

    Panel d = ts_delta(x, 2);
    assert(close(d(4, 0), 2.f) && std::isnan(d(5, 1)));

    Panel r(1, 4);
    r(0, 0) = 3; r(0, 1) = 1; r(0, 2) = kNaN; r(0, 3) = 1;
    Panel rr = rank(r);
    assert(close(rr(0, 0), 1.f) && close(rr(0, 1), 0.25f) && close(rr(0, 3), 0.25f) && std::isnan(rr(0, 2)));

    std::puts("cpp ops: ok");
}
