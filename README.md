# Sutra

Alpha research engine for Indian equities. You write formulaic alphas such as

```
rank(-ts_corr(close, volume, 10))
group_neutralize(ts_decay_linear(ts_delta(close, 5) / ts_std(close, 20), 8))
```

and Sutra evaluates them over a point-in-time panel of NSE stocks, scores them
under purged walk-forward validation, and combines the survivors into a
market-neutral portfolio with realistic Indian trading costs.

> Status: work in progress. The expression language, reference operators,
> backtester, purged CV, Deflated Sharpe Ratio and the first C++ kernels work and
> are tested. Data pipeline, GP search, signal combination and the optimiser are
> being built now. Results will go here once they come from out-of-sample runs.

## Design

**Expression engine.** Expressions parse into a hashable AST (`python/sutra/expr.py`),
so identical sub-trees are evaluated once and duplicate candidates are dropped.
Operators come in two implementations: a NumPy reference (`python/sutra/ops.py`)
and C++20 kernels over column-major float panels (`cpp/`), exposed through
pybind11. The C++ path is differentially tested against the reference on random
panels with missing data.

**Point-in-time data.** The universe is rebuilt from NSE daily bhavcopies, so
stocks that later delisted stay in the history (no survivorship bias). Stocks are
keyed on ISIN because symbols change. Splits and bonuses are detected on the
ex-date from NSE's restated previous close.

**Research hygiene, enforced in code.**
- Purged walk-forward splits with an embargo (`python/sutra/cv.py`).
- Deflated Sharpe Ratio (Bailey and Lopez de Prado) counting every candidate tried,
  including rejected ones (`python/sutra/metrics.py`).
- A look-ahead canary in the test suite: on random returns a legitimate signal must
  earn nothing, and the same engine fed a look-ahead signal must look absurd.

**Portfolio.** Dollar-, sector- and beta-neutral mean-variance optimisation with a
turnover penalty (`python/sutra/portfolio.py`, cvxpy + OSQP). Shorts are limited
to F&O stocks, since in India the short leg runs through stock futures.

## Layout

```
python/sutra/   expr.py ops.py backtest.py cv.py metrics.py portfolio.py
                data/bhavcopy.py   search/gp.py
cpp/            include/sutra/{panel,ops}.hpp  src/{ops,bindings}.cpp  tests/
tests/          Python tests (pytest)
```

## Quick start

```bash
pip install numpy scipy pandas pytest pybind11      # + cvxpy lightgbm polars for later stages
PYTHONPATH=python pytest -q tests

cmake -S . -B build && cmake --build build -j && ./build/test_ops
PYTHONPATH=python:build python -c "import _core"    # C++ kernels from Python
```

## Roadmap

- [x] Expression parser and NumPy reference operators
- [x] Daily backtester with explicit timing, look-ahead canary
- [x] Purged walk-forward CV, Deflated Sharpe Ratio
- [x] C++ kernels (ts_mean, ts_delta, rank) + pybind11, differential tests
- [ ] Bhavcopy download and point-in-time panel (Parquet), corporate actions
- [ ] Remaining operators in C++, AVX-2 rolling windows, benchmarks
- [ ] Genetic-programming search with correlation cap against the alpha pool
- [ ] Signal combination (ridge, LightGBM on rank-transformed signals)
- [ ] Optimiser with sector/beta neutrality, F&O-only shorts, full Indian cost model
- [ ] Out-of-sample report
