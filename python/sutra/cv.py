"""Purged walk-forward splits.

A label at date t looks `horizon` days ahead, so training rows within
`horizon` days of the test start share information with the test set. We drop
them (purge) plus an extra `embargo` days for serial correlation.
"""


def purged_walk_forward(n_dates, n_splits, test_size, horizon=1, embargo=10, min_train=252):
    """Yield (train_range, test_range) as Python ranges over date indices.
    Expanding window; tests are the last n_splits * test_size dates."""
    first_test = n_dates - n_splits * test_size
    if first_test - horizon - embargo < min_train:
        raise ValueError("not enough history for the requested splits")
    for k in range(n_splits):
        start = first_test + k * test_size
        yield range(0, start - horizon - embargo), range(start, start + test_size)
