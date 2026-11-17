"""Market-neutral portfolio construction (one date at a time).

maximise   alpha'w - risk_aversion * w' S w - tcost * |w - w_prev|_1
subject to sum(w) = 0                (dollar neutral)
           B' w = 0                  (sector dummies and beta columns in B)
           |w|_1 <= gross, |w_i| <= cap
           w_i >= 0 where shortable is False   (India: only F&O names can be shorted, via futures)
"""
import numpy as np


def optimise(alpha, cov, w_prev, exposures, shortable, risk_aversion=10.0,
             tcost=10e-4, gross=1.0, cap=0.02):
    import cvxpy as cp  # optional dependency

    n = len(alpha)
    w = cp.Variable(n)
    objective = cp.Maximize(
        alpha @ w
        - risk_aversion * cp.quad_form(w, cp.psd_wrap(cov))
        - tcost * cp.norm1(w - w_prev)
    )
    cons = [cp.sum(w) == 0, exposures.T @ w == 0, cp.norm1(w) <= gross, cp.abs(w) <= cap]
    long_only = np.flatnonzero(~np.asarray(shortable, bool))
    if long_only.size:
        cons.append(w[long_only] >= 0)
    cp.Problem(objective, cons).solve(solver=cp.OSQP)
    return np.asarray(w.value).ravel()
