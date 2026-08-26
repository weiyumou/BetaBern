"""
Data Splitter for Knowledge Tracing

This module provides flexible data splitting strategies for KT experiments:
- "within": Sequential split per user (for temporal evaluation)
- "between": Strict global user holdout (for cold start)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    TimeSeriesSplit,
)


class DataSplitter:
    """
    Flexible data splitter for knowledge tracing experiments.

    Parameters
    ----------
    strategy : str
        Splitting strategy (supported by both ``split`` (K-fold) and ``train_test_split``):
        - "within": Sequential (temporal) split per (user, skill)
        - "within_random": Random per-(user, skill) item holdout (for static-trait measurement)
        - "between": Strict global user holdout (cold start)
    n_splits : int
        Number of cross-validation folds for ``split``. (default is 3)
    random_state : int
        Random seed for reproducibility (fold assignment in ``split``; the held-out subset in
        ``train_test_split`` takes its own ``random_state`` argument).
    """

    USER_COL = "user_id"
    SKILL_COL = "skill_name"

    def __init__(self, strategy: str, n_splits: int = 3, random_state: int = 42):
        self.strategy = strategy
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Generate train/test indices for cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            The data to split.
        y : ignored
            Not used, present for API compatibility.
        groups : ignored
            Not used, present for API compatibility.

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train and test indices for each fold.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")
        assert y is None and groups is None, "y and groups are not used in this splitter."

        match self.strategy:
            case "within":
                yield from self._split_within(X)
            case "within_random":
                yield from self._split_within_random(X)
            case "between":
                yield from self._split_between(X)
            case _:
                raise ValueError(f"Unknown strategy: {self.strategy}")

    def _split_within(self, df: pd.DataFrame):
        """Sequential split per user."""
        fold_buckets = [{"train": [], "test": []} for _ in range(self.n_splits)]
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # Group the INTEGER INDICES by user
        indices = np.arange(len(df))

        # group is a Series of row indices for this user
        for _, group in pd.Series(indices).groupby([df[self.USER_COL].to_numpy(), df[self.SKILL_COL].to_numpy()]):

            if len(group) < self.n_splits + 1:
                warnings.warn(
                    "Some (user, skill) groups have fewer interactions than n_splits + 1. "
                    "They will be skipped in 'within' strategy."
                )
                continue

            # Map Series back to numpy array for splitting
            row_indices = group.to_numpy()

            # tscv.split returns indices relative to the group array
            for fold_i, (local_train, local_test) in enumerate(tscv.split(row_indices)):
                # Map local relative pos -> Global integer pos
                fold_buckets[fold_i]["train"].append(row_indices[local_train])
                fold_buckets[fold_i]["test"].append(row_indices[local_test])

        for fold_i in range(self.n_splits):
            if not fold_buckets[fold_i]["train"]:
                raise ValueError(
                    "No valid user-skill pairs found for 'within' splitting. "
                    "Check your data and n_splits setting."
                )

            yield (
                np.sort(np.concatenate(fold_buckets[fold_i]["train"])),
                np.sort(np.concatenate(fold_buckets[fold_i]["test"])),
            )

    def _split_within_random(self, df: pd.DataFrame):
        """Random per-(user, skill) K-fold.

        Each group's rows are randomly partitioned into ``n_splits`` folds, so every fold holds out a
        random subset of each student's items while the rest stay trainable (θ remains estimable). The
        right within-student protocol for a static trait, where response order is meaningless.
        """
        rng = np.random.default_rng(self.random_state)
        df_temp = pd.DataFrame({
            "u": df[self.USER_COL].to_numpy(),
            "s": df[self.SKILL_COL].to_numpy(),
            "key": rng.random(len(df)),
        })
        gb = df_temp.groupby(["u", "s"])
        rank = gb["key"].rank(method="first").to_numpy().astype(int) - 1  # random 0-based rank in group
        # A random per-group offset rotates fold labels so the size remainder (group_size % n_splits)
        # lands on a different fold per group, keeping fold sizes balanced rather than loading fold 0.
        group_ids = gb.ngroup().to_numpy()
        offsets = rng.integers(0, self.n_splits, size=group_ids.max() + 1)
        fold_ids = (rank + offsets[group_ids]) % self.n_splits
        indices = np.arange(len(df))
        for fold_i in range(self.n_splits):
            test_mask = fold_ids == fold_i
            yield np.sort(indices[~test_mask]), np.sort(indices[test_mask])

    def _split_between(self, df: pd.DataFrame):
        """Strict global user holdout via sklearn's ``GroupKFold``: whole students are partitioned into
        ``n_splits`` disjoint test folds (shuffled + seeded), so each fold scores a held-out set of
        students absent from its training data (cold-start evaluation). GroupKFold balances the number
        of *responses* per fold (vs. equal users per fold), and requires ``n_splits <= n_users``."""
        gkf = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, test_idx in gkf.split(df, groups=df[self.USER_COL].to_numpy()):
            yield np.sort(train_idx), np.sort(test_idx)

    def train_test_split(self, data: pd.DataFrame, test_size: float, random_state: int | None = None):
        """
        Split a DataFrame into train and test subsets according to the strategy.

        Parameters
        ----------
        data : pd.DataFrame
            The data to split.
        test_size : float
            Proportion of data to use for testing. (default is 0.1)
        random_state : int
            Random seed for reproducibility (used in "between" strategy).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Train and test indices for the split.
        """
        # We work with indices relative to this new subset X_train_full
        subset_indices = np.arange(len(data))

        if self.strategy == "within":
            # Manual sequential cut (vectorized)
            users = data[self.USER_COL].to_numpy()
            skills = data[self.SKILL_COL].to_numpy()

            # pandas groupby for ranking
            df_temp = pd.DataFrame({"u": users, "s": skills, "idx": subset_indices})
            gb = df_temp.groupby(["u", "s"])
            df_temp["count"] = gb["u"].transform("count")
            df_temp["rank"] = gb.cumcount()

            cutoff = df_temp["count"] * (1 - test_size)
            test_mask = df_temp["rank"] >= cutoff

            train_idx = subset_indices[~test_mask]
            test_idx = subset_indices[test_mask]
            return train_idx, test_idx

        elif self.strategy == "within_random":
            # Random per-(user, skill) item holdout: same per-group proportions as "within", but the
            # held-out rows are a random subset rather than the temporal tail — the right protocol for
            # a static latent trait, where response order is meaningless.
            df_temp = pd.DataFrame({
                "u": data[self.USER_COL].to_numpy(),
                "s": data[self.SKILL_COL].to_numpy(),
                "key": np.random.default_rng(random_state).random(len(data)),
            })
            gb = df_temp.groupby(["u", "s"])
            df_temp["count"] = gb["u"].transform("count")
            df_temp["rank"] = gb["key"].rank(method="first") - 1  # random 0-based rank within group
            count = df_temp["count"].to_numpy()
            # Hold out ~count*test_size per group, but >=1 for any group with >=2 rows so the split is
            # never empty on small groups (e.g. a per-fold validation carve); never hold out a lone row.
            cutoff = np.where(count >= 2, np.minimum(count * (1 - test_size), count - 1), count.astype(float))
            test_mask = df_temp["rank"].to_numpy() >= cutoff
            return subset_indices[~test_mask], subset_indices[test_mask]

        else:
            # Group Shuffle for "Between" strategies
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            groups = data[self.USER_COL].to_numpy()

            train_idx, test_idx = next(gss.split(subset_indices, groups=groups))

            # Sort indices to preserve order
            return np.sort(train_idx), np.sort(test_idx)
