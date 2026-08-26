"""Tests for the K-fold cross-validation generators in DataSplitter (prediction protocols)."""
import numpy as np
import pandas as pd
import pytest

from betabern.core.data.splitter import DataSplitter

N_USERS, N_SKILLS, N_ITEMS, K = 15, 2, 8, 5


@pytest.fixture
def df():
    rows = [{"user_id": u, "skill_name": s, "item_id": i}
            for u in range(N_USERS) for s in range(N_SKILLS) for i in range(N_ITEMS)]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy", ["within_random", "between"])
def test_kfold_is_a_partition(df, strategy):
    folds = list(DataSplitter(strategy=strategy, n_splits=K, random_state=0).split(df))
    n = len(df)
    assert len(folds) == K
    tested = np.zeros(n, dtype=int)
    for train, test in folds:
        assert set(train.tolist()).isdisjoint(test.tolist())  # no leakage within a fold
        assert len(train) + len(test) == n  # train/test cover everything
        tested[test] += 1
    assert (tested == 1).all()  # every row held out exactly once


def test_within_random_keeps_users_trainable_and_balanced(df):
    folds = list(DataSplitter(strategy="within_random", n_splits=K, random_state=0).split(df))
    all_users = set(df["user_id"])
    for train, _ in folds:
        assert set(df.iloc[train]["user_id"]) == all_users  # θ estimable for every student each fold
    sizes = [len(test) for _, test in folds]
    assert max(sizes) - min(sizes) <= N_USERS * N_SKILLS  # remainder spread, not dumped on one fold


def test_between_is_user_disjoint(df):
    folds = list(DataSplitter(strategy="between", n_splits=K, random_state=0).split(df))
    for train, test in folds:
        assert set(df.iloc[train]["user_id"]).isdisjoint(df.iloc[test]["user_id"])  # cold-start holdout


@pytest.mark.parametrize("strategy", ["within_random", "between"])
def test_kfold_is_deterministic(df, strategy):
    a = [test.tolist() for _, test in DataSplitter(strategy, n_splits=K, random_state=0).split(df)]
    b = [test.tolist() for _, test in DataSplitter(strategy, n_splits=K, random_state=0).split(df)]
    assert a == b
