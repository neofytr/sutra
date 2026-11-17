"""Performance and research-hygiene metrics."""
import numpy as np
from scipy.stats import kurtosis, norm, skew

from .ops import rank

EULER_GAMMA = 0.5772156649015329


def sharpe(daily, periods=252):
    daily = np.asarray(daily, float)
    daily = daily[~np.isnan(daily)]
    return np.sqrt(periods) * daily.mean() / daily.std(ddof=1)


def daily_ic(signal, fwd_ret):
    """Spearman IC per date: Pearson correlation of cross-sectional ranks."""
    rs, rr = rank(signal), rank(fwd_ret)
    out = np.full(signal.shape[0], np.nan)
    for t in range(signal.shape[0]):
        ok = ~np.isnan(rs[t]) & ~np.isnan(rr[t])
        if ok.sum() > 2:
            out[t] = np.corrcoef(rs[t, ok], rr[t, ok])[0, 1]
    return out


def max_drawdown(daily):
    eq = np.nancumsum(daily)
    return float(np.max(np.maximum.accumulate(eq) - eq))


def expected_max_sharpe(sr_variance, n_trials):
    """E[max SR] over n_trials independent trials with true SR 0
    (Bailey & Lopez de Prado 2014). Per-period (not annualised) units."""
    if n_trials < 2:
        return 0.0
    g = EULER_GAMMA
    return np.sqrt(sr_variance) * (
        (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e))
    )


def deflated_sharpe(daily, sr_variance, n_trials):
    """Probability that the true Sharpe of `daily` beats the best Sharpe you would
    expect from n_trials pure-noise strategies. sr_variance: variance of the
    per-period Sharpe across all trials you ran. Report this, not the raw Sharpe."""
    r = np.asarray(daily, float)
    r = r[~np.isnan(r)]
    sr = r.mean() / r.std(ddof=1)
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom))
