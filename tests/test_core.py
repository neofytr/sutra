import numpy as np
import pytest

from sutra import backtest, cv, metrics, ops
from sutra.expr import evaluate, parse

rng = np.random.default_rng(0)


def test_ts_mean_matches_manual():
    x = rng.normal(size=(30, 4))
    out = ops.ts_mean(x, 5)
    assert np.isnan(out[:4]).all()
    assert np.allclose(out[10], x[6:11].mean(axis=0))


def test_ts_window_with_nan_is_nan():
    x = rng.normal(size=(20, 2))
    x[10, 0] = np.nan
    out = ops.ts_mean(x, 3)
    assert np.isnan(out[10:13, 0]).all() and not np.isnan(out[13, 0])


def test_rank_handles_nan_and_ties():
    x = np.array([[3.0, 1.0, np.nan, 1.0]])
    r = ops.rank(x)
    assert np.isnan(r[0, 2])
    assert r[0, 0] == 1.0 and r[0, 1] == r[0, 3] == 0.25


def test_ts_corr_of_series_with_itself_is_one():
    x = rng.normal(size=(40, 3))
    assert np.allclose(ops.ts_corr(x, x, 10)[9:], 1.0)


def test_parse_and_evaluate():
    close = rng.lognormal(size=(50, 6))
    volume = rng.lognormal(size=(50, 6))
    ast = parse("rank(-ts_corr(close, volume, 10)) - 0.5")
    out = evaluate(ast, {"close": close, "volume": volume}, cache={})
    ref = ops.rank(-ops.ts_corr(close, volume, 10)) - 0.5
    assert np.allclose(out, ref, equal_nan=True)


def test_parse_errors():
    with pytest.raises(SyntaxError):
        parse("rank(close")
    with pytest.raises(SyntaxError):
        parse("not_an_op(close)")


def test_purged_walk_forward_leaves_gap():
    for train, test in cv.purged_walk_forward(2000, 4, 250, horizon=5, embargo=10):
        assert test.start - train.stop == 15
        assert len(test) == 250


def test_lookahead_canary():
    """On iid random returns, a signal using only data up to today's close
    (today's return) must earn nothing. If the engine ever lets day-t weights
    earn day-t returns, this signal shows an absurd Sharpe."""
    T, N = 1500, 200
    r = rng.normal(0, 0.02, size=(T, N))
    res = backtest.run(r, r, delay=1, cost_bps=0)
    assert abs(metrics.sharpe(res["pnl"])) < 1.0
    # the same engine fed a look-ahead signal (tomorrow's return) is absurdly good,
    # so a too-good backtest is a bug in the signal, not luck
    cheat = np.vstack([r[1:], np.full((1, N), np.nan)])   # cheat[t] = r[t+1]
    assert metrics.sharpe(backtest.run(cheat, r, delay=1, cost_bps=0)["pnl"]) > 20


def test_weights_are_dollar_neutral():
    w = backtest.signal_to_weights(rng.normal(size=(10, 50)))
    assert np.allclose(w.sum(axis=1), 0) and np.allclose(np.abs(w).sum(axis=1), 1)


def test_deflated_sharpe_penalises_many_trials():
    daily = rng.normal(0.001, 0.01, size=1000)
    one = metrics.deflated_sharpe(daily, sr_variance=0.002, n_trials=1)
    many = metrics.deflated_sharpe(daily, sr_variance=0.002, n_trials=10_000)
    assert many < one
