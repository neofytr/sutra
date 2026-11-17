"""Vectorised daily backtest for cross-sectional signals.

Timing convention (the place look-ahead bugs live):
  signal[t] uses data up to the close of day t.
  With delay=1 the weights from signal[t] are held over day t+1, i.e. we trade
  at the close of t and earn the close-to-close return from t to t+1.
  A larger delay waits delay-1 more days before trading.
"""
import warnings

import numpy as np


def signal_to_weights(signal, gross=1.0):
    """Dollar-neutral: demean across stocks, scale to the target gross exposure."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        w = signal - np.nanmean(signal, axis=1, keepdims=True)
    g = np.nansum(np.abs(w), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(g > 0, w / g * gross, 0.0)
    return np.nan_to_num(w)


def run(signal, returns, delay=1, cost_bps=10.0):
    """returns[t] = close-to-close return from t-1 to t.
    Returns dict of daily net pnl, gross pnl, turnover (all length T)."""
    w = signal_to_weights(signal)
    held = np.zeros_like(w)
    held[delay:] = w[:-delay]          # weights decided on t-delay, held over day t
    r = np.nan_to_num(returns)
    gross_pnl = (held * r).sum(axis=1)
    turnover = np.abs(np.diff(held, axis=0, prepend=0.0)).sum(axis=1)
    net = gross_pnl - turnover * cost_bps * 1e-4
    return {"pnl": net, "gross": gross_pnl, "turnover": turnover}
