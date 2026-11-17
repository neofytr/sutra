#include "sutra/ops.hpp"

#include <algorithm>
#include <numeric>

namespace sutra {

Panel ts_mean(const Panel& x, int d) {
    Panel out(x.T, x.N);
    for (std::size_t j = 0; j < x.N; ++j) {
        const float* in = x.col(j);
        float* o = out.col(j);
        // running sum; restart the window after a NaN so NaN never poisons the sum
        double sum = 0.0;
        int valid = 0;
        for (std::size_t t = 0; t < x.T; ++t) {
            if (std::isnan(in[t])) { sum = 0.0; valid = 0; continue; }
            sum += in[t];
            if (++valid > d) { sum -= in[t - d]; valid = d; }
            if (valid == d) o[t] = static_cast<float>(sum / d);
        }
    }
    return out;
}

Panel ts_delta(const Panel& x, int d) {
    Panel out(x.T, x.N);
    for (std::size_t j = 0; j < x.N; ++j)
        for (std::size_t t = static_cast<std::size_t>(d); t < x.T; ++t)
            out(t, j) = x(t, j) - x(t - d, j);  // NaN propagates naturally
    return out;
}

Panel rank(const Panel& x) {
    Panel out(x.T, x.N);
    std::vector<std::size_t> idx;
    for (std::size_t t = 0; t < x.T; ++t) {
        idx.clear();
        for (std::size_t j = 0; j < x.N; ++j)
            if (!std::isnan(x(t, j))) idx.push_back(j);
        const std::size_t n = idx.size();
        if (n < 2) continue;
        std::sort(idx.begin(), idx.end(), [&](auto a, auto b) { return x(t, a) < x(t, b); });
        for (std::size_t i = 0; i < n;) {  // average rank over runs of ties
            std::size_t k = i;
            while (k + 1 < n && x(t, idx[k + 1]) == x(t, idx[i])) ++k;
            const float r = static_cast<float>((i + k) / 2.0 / (n - 1));
            for (std::size_t m = i; m <= k; ++m) out(t, idx[m]) = r;
            i = k + 1;
        }
    }
    return out;
}

}  // namespace sutra
