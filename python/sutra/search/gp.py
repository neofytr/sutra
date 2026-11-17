"""Genetic-programming search over alpha expressions. Not written yet.

Plan:
  - population of ASTs from sutra.expr (tuples, so they hash and dedupe)
  - mutation: swap an operator, perturb a window, replace a subtree
  - crossover: swap random subtrees between two parents
  - fitness on the TRAIN folds only (sutra.cv.purged_walk_forward): mean IC,
    penalised by turnover and by max |corr| to alphas already in the pool
  - every evaluated candidate is counted toward n_trials for
    sutra.metrics.deflated_sharpe, including the rejected ones
"""
