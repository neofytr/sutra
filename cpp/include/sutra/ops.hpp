#pragma once
#include "sutra/panel.hpp"

namespace sutra {

// Scalar reference kernels. Same semantics as python/sutra/ops.py
// (a window containing any NaN gives NaN). AVX-2 versions go next to these
// and are differentially tested against them.
Panel ts_mean(const Panel& x, int d);
Panel ts_delta(const Panel& x, int d);
Panel rank(const Panel& x);  // cross-sectional percentile in [0, 1], ties averaged

}  // namespace sutra
