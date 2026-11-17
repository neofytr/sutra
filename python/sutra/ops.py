"""Reference (NumPy) implementations of the alpha operators.

Every panel is a float array of shape (T, N): T dates (rows) by N stocks
(columns). NaN means "no data" (not listed yet, delisted, suspended).

These are the correctness oracle for the C++ kernels in cpp/: the C++ path
is differentially tested against these functions, so keep them simple.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(x, d):
    """(T, N) -> (T, N, d) view; the first d-1 rows are padded with NaN."""
    pad = np.full((d - 1, x.shape[1]), np.nan)
    return sliding_window_view(np.vstack([pad, x]), d, axis=0)


def _full(w):
    """Mask of windows with no NaN (we require a full history, no partial means)."""
    return ~np.isnan(w).any(axis=-1)


# ---------- time-series operators (per stock, along dates) ----------

def ts_delay(x, d):
    out = np.full_like(x, np.nan)
    out[d:] = x[:-d]
    return out


def ts_delta(x, d):
    return x - ts_delay(x, d)


def ts_sum(x, d):
    w = _windows(x, d)
    return np.where(_full(w), w.sum(axis=-1), np.nan)


def ts_mean(x, d):
    return ts_sum(x, d) / d


def ts_std(x, d):
    w = _windows(x, d)
    return np.where(_full(w), w.std(axis=-1, ddof=1), np.nan)


def ts_min(x, d):
    w = _windows(x, d)
    return np.where(_full(w), w.min(axis=-1), np.nan)


def ts_max(x, d):
    w = _windows(x, d)
    return np.where(_full(w), w.max(axis=-1), np.nan)


def ts_rank(x, d):
    """Percentile of today's value within its own last d values, in [0, 1]."""
    w = _windows(x, d)
    last = w[..., -1:]
    r = (w < last).sum(axis=-1) + 0.5 * ((w == last).sum(axis=-1) - 1)
    return np.where(_full(w), r / (d - 1), np.nan)


def ts_decay_linear(x, d):
    """Linearly weighted mean, newest value weighted d, oldest weighted 1."""
    wts = np.arange(1, d + 1, dtype=float)
    w = _windows(x, d)
    return np.where(_full(w), (w * wts).sum(axis=-1) / wts.sum(), np.nan)


def ts_corr(x, y, d):
    wx, wy = _windows(x, d), _windows(y, d)
    ok = _full(wx) & _full(wy)
    mx, my = wx.mean(axis=-1, keepdims=True), wy.mean(axis=-1, keepdims=True)
    cov = ((wx - mx) * (wy - my)).sum(axis=-1)
    den = np.sqrt(((wx - mx) ** 2).sum(axis=-1) * ((wy - my) ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok & (den > 0), cov / den, np.nan)


# ---------- cross-sectional operators (per date, across stocks) ----------

def rank(x):
    """Cross-sectional percentile rank in [0, 1], NaNs stay NaN, ties averaged."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        row = x[t]
        ok = ~np.isnan(row)
        n = ok.sum()
        if n < 2:
            continue
        v = row[ok]
        order = v.argsort(kind="mergesort")
        r = np.empty(n)
        r[order] = np.arange(n)
        # average ranks over ties
        _, inv = np.unique(v, return_inverse=True)
        r = np.bincount(inv, weights=r)[inv] / np.bincount(inv)[inv]
        out[t, ok] = r / (n - 1)
    return out


def zscore(x):
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - m) / s


def demean(x):
    return x - np.nanmean(x, axis=1, keepdims=True)


def group_neutralize(x, groups):
    """Demean within each group (e.g. sector) on every date. groups: (N,) ints."""
    out = np.array(x, dtype=float)
    for g in np.unique(groups):
        cols = groups == g
        out[:, cols] -= np.nanmean(x[:, cols], axis=1, keepdims=True)
    return out


# ---------- elementwise ----------

def sign(x):
    return np.sign(x)


def log(x):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log(x)


def abs_(x):
    return np.abs(x)


OPS = {
    "ts_delay": ts_delay, "ts_delta": ts_delta, "ts_sum": ts_sum,
    "ts_mean": ts_mean, "ts_std": ts_std, "ts_min": ts_min, "ts_max": ts_max,
    "ts_rank": ts_rank, "ts_decay_linear": ts_decay_linear, "ts_corr": ts_corr,
    "rank": rank, "zscore": zscore, "demean": demean,
    "group_neutralize": group_neutralize,
    "sign": sign, "log": log, "abs": abs_,
}
