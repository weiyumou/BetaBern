"""
Data Loading and Dataset Classes for Knowledge Tracing

This module provides:
- Data loading utilities
- PyTorch Dataset classes for user activity sequences
- Lightning DataModule for training pipelines
"""

import math
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, StackDataset

from betabern.core.data.splitter import DataSplitter

# Column name constants
USER_COL = "user_id"
ITEM_COL = "item_id"
SKILL_COL = "skill_name"
ANSWER_COL = "is_correct"
TIME_COL = "timestamp"
DELTA_COL = "time_delta"

PAD_VALUE = -1


@dataclass
class DatasetStats:
    num_users: int
    num_skills: int
    num_items: int
    num_train: int
    num_val: int
    num_test: int
    total_interactions: int


def is_stacked_batch(batch) -> bool:
    """
    Check if the batch is a stacked batch (history + target).
    """
    return (
            isinstance(batch, (tuple, list))
            and len(batch) == 2
            and isinstance(batch[0], dict)
            and isinstance(batch[1], dict)
    )


def load_data(data_path: str, skills: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Load student activity data from a CSV file.

    The CSV file should have at least the following columns:
        user_id | skill_name | item_id | is_correct | timestamp

    Parameters
    ----------
    data_path : str
        Path to a CSV file
    skills : list[str], optional
        If provided, filter to only include these skills.

    Returns
    -------
    pd.DataFrame
        Loaded and preprocessed DataFrame.
    """

    df = pd.read_csv(data_path, parse_dates=[TIME_COL])
    df = df.sort_values(by=[USER_COL, TIME_COL]).reset_index(drop=True)

    if skills is not None:
        df = df[df[SKILL_COL].isin(skills)].copy()
    return df


class KnowledgeTracingDataset(Dataset):
    """
    Dataset for user activity sequences.

    Each sample is a (user, skill) pair's complete interaction sequence.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with user activity data.
    """

    def __init__(self, df: pd.DataFrame, user_skill_pairs: Optional[Sequence[tuple]] = None):
        self.df = df.copy()

        if user_skill_pairs is not None:
            self.user_skill_pairs = user_skill_pairs
        else:
            self.user_skill_pairs = list(df.groupby([USER_COL, SKILL_COL], sort=True).groups.keys())

    def __getitem__(self, idx: int):
        user_idx, skill_idx = self.user_skill_pairs[idx]
        grp = self.df[self.df[USER_COL].eq(user_idx) & self.df[SKILL_COL].eq(skill_idx)]

        item_ids = torch.as_tensor(grp[ITEM_COL].to_numpy(), dtype=torch.long)
        answers = torch.as_tensor(grp[ANSWER_COL].to_numpy(), dtype=torch.long)
        time_deltas = torch.as_tensor(grp[DELTA_COL].to_numpy(), dtype=torch.float)

        return {
            "user_id": user_idx,
            "skill_id": skill_idx,
            "item_ids": item_ids,
            "answers": answers,
            "time_deltas": time_deltas,
        }

    def __len__(self):
        return len(self.user_skill_pairs)


class KnowledgeTracingDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Knowledge Tracing experiments.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input DataFrame.
    train_inds : Sequence, optional
        Pre-split training indices.
    test_inds : Sequence, optional
        Pre-split test indices.
    train_val_test_split : tuple
        Proportions for train/val/test split. (default is (0.7, 0.15, 0.15))
    split_strategy : str
        Splitting strategy: "within" (temporal per-(user, skill) tail holdout), "within_random"
        (random per-(user, skill) item holdout — for static-trait measurement), or "between"
        (cold-start global user holdout). (default is "within")
    batch_size : int
        Batch size for DataLoaders. (default is 64)
    num_workers : int
        Number of DataLoader workers. (default is 0)
    """

    def __init__(
            self,
            df: pd.DataFrame,
            train_inds: Optional[Sequence] = None,
            test_inds: Optional[Sequence] = None,
            train_val_test_split: tuple = (0.7, 0.15, 0.15),
            split_strategy: str = "within",
            batch_size: int = 64,
            num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["df", "train_inds", "test_inds"])

        # Compute time delta between consecutive interactions per user-skill pair
        df[DELTA_COL] = df.groupby([USER_COL, SKILL_COL])[TIME_COL].diff().dt.total_seconds().fillna(0)
        df[DELTA_COL] = np.log1p(df[DELTA_COL] / 3600 / 24)  # convert to log(1 + days)

        # Encode the full DataFrame
        self.df, self.maps = self.encode_data(df)

        train_sz, val_sz, test_sz = train_val_test_split
        splitter = DataSplitter(strategy=split_strategy, n_splits=1)
        if math.isclose(test_sz, 0.0):
            self.train_df, self.test_df = self.df, None
        else:
            # If train and test indices are not given, perform the split
            if train_inds is None and test_inds is None:
                train_inds, test_inds = splitter.train_test_split(self.df, test_size=test_sz, random_state=0)
            self.train_df = self.df.iloc[train_inds]
            self.test_df = self.df.iloc[test_inds]

        if math.isclose(val_sz, 0.0):
            self.val_df = None
        else:
            # Split the train DataFrame into train and validation
            train_inds, val_inds = splitter.train_test_split(self.train_df,
                                                             test_size=val_sz / (train_sz + val_sz),
                                                             random_state=42)
            self.val_df = self.train_df.iloc[val_inds]
            self.train_df = self.train_df.iloc[train_inds]

        # Add split markers in the full DataFrame
        self.df["split"] = None
        self.df.loc[self.train_df.index, "split"] = "train"
        if self.val_df is not None:
            self.df.loc[self.val_df.index, "split"] = "val"
        if self.test_df is not None:
            self.df.loc[self.test_df.index, "split"] = "test"

        self.items_to_skills = self.get_items_to_skills()
        self.split_strategy = split_strategy
        self.batch_size, self.num_workers = batch_size, num_workers
        self.train_ds, self.val_ds, self.test_ds = (None,) * 3
        self.val_collate_fn, self.test_collate_fn = None, None

        self.predict_ds = None

    @staticmethod
    def encode_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Encode User, Skill, and Item columns to integer indices.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        tuple[pd.DataFrame, dict]
            Encoded DataFrame and dictionary of LabelEncoders.
        """
        le_user = LabelEncoder()
        le_skill = LabelEncoder()
        le_item = LabelEncoder()

        df = df.copy()
        df[USER_COL] = le_user.fit_transform(df[USER_COL])
        df[SKILL_COL] = le_skill.fit_transform(df[SKILL_COL])
        df[ITEM_COL] = le_item.fit_transform(df[ITEM_COL])

        maps = {
            "users": le_user,
            "skills": le_skill,
            "items": le_item,
        }
        return df, maps

    @cached_property
    def stats(self) -> DatasetStats:
        return DatasetStats(num_users=len(self.maps["users"].classes_) if "users" in self.maps else 0,
                            num_skills=len(self.maps["skills"].classes_) if "skills" in self.maps else 0,
                            num_items=len(self.maps["items"].classes_) if "items" in self.maps else 0,
                            num_train=len(self.train_df),
                            num_val=len(self.val_df) if self.val_df is not None else 0,
                            num_test=len(self.test_df) if self.test_df is not None else 0,
                            total_interactions=len(self.df))

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training/validation/testing/prediction."""

        self.train_ds = KnowledgeTracingDataset(self.train_df)
        history_df = self.train_df

        if (self.val_df is not None) and (stage in {"fit", "validate"}):
            val_ds = KnowledgeTracingDataset(self.val_df)
            if self.split_strategy in ("within", "within_random"):
                if not set(val_ds.user_skill_pairs).issubset(self.train_ds.user_skill_pairs):
                    warnings.warn("Some unseen (user, skill) pairs are in the validation set.")
                # Warm start: use history + target
                self.val_ds = StackDataset(KnowledgeTracingDataset(self.train_df, val_ds.user_skill_pairs), val_ds)
                self.val_collate_fn = self.collate_fn_stack
            elif self.split_strategy == "between":
                if not set(val_ds.user_skill_pairs).isdisjoint(self.train_ds.user_skill_pairs):
                    raise ValueError("Some (user, skill) pairs in the training set are leaked to the validation set.")
                self.val_ds = val_ds
                self.val_collate_fn = self.collate_fn
            history_df = pd.concat([self.train_df, self.val_df], ignore_index=True)

        if (self.test_df is not None) and (stage == "test"):
            test_ds = KnowledgeTracingDataset(self.test_df)
            if self.split_strategy in ("within", "within_random"):
                # Warm start: use history + target
                if not set(test_ds.user_skill_pairs).issubset(self.train_ds.user_skill_pairs):
                    warnings.warn("Some unseen (user, skill) pairs are in the test set.")
                self.test_ds = StackDataset(KnowledgeTracingDataset(history_df, test_ds.user_skill_pairs), test_ds)
                self.test_collate_fn = self.collate_fn_stack
            elif self.split_strategy == "between":
                val_pairs = set() if self.val_df is None else set(KnowledgeTracingDataset(self.val_df).user_skill_pairs)
                if not set(test_ds.user_skill_pairs).isdisjoint(self.train_ds.user_skill_pairs):
                    raise ValueError("Some (user, skill) pairs in the training set are leaked to the test set.")
                if not set(test_ds.user_skill_pairs).isdisjoint(val_pairs):
                    raise ValueError("Some (user, skill) pairs in the validation set are leaked to the test set.")
                self.test_ds = test_ds
                self.test_collate_fn = self.collate_fn

        if stage == "predict":
            self.predict_ds = KnowledgeTracingDataset(self.df)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def val_dataloader(self):
        if not self.val_ds:
            return []
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.val_collate_fn, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def test_dataloader(self):
        if not self.test_ds:
            return []
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.test_collate_fn, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def predict_dataloader(self):
        if not self.predict_ds:
            return []
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    def recovery_dataloader(self):
        """Self-stacked ``(full, full)`` batches over every (user, skill): each pair's complete response
        vector is both the history and the target, so a task's ``predict_step`` stacked branch scores it
        by posterior-predictive / EAP from its own responses — the ground-truth recovery pass, with no
        model-specific scoring code. Needs no ``setup`` (it is self-contained)."""
        ds = KnowledgeTracingDataset(self.df)
        return DataLoader(StackDataset(ds, ds), batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn_stack, num_workers=self.num_workers,
                          pin_memory=torch.cuda.is_available())

    @staticmethod
    def collate_fn(x, pad_value: int = PAD_VALUE):
        """Collate function that pads sequences to max length in batch."""
        user_ids = torch.tensor([s["user_id"] for s in x], dtype=torch.long)
        skill_ids = torch.tensor([s["skill_id"] for s in x], dtype=torch.long)

        padded_item_ids = pad_sequence(
            [s["item_ids"] for s in x], batch_first=True, padding_value=pad_value
        )
        padded_answers = pad_sequence(
            [s["answers"] for s in x], batch_first=True, padding_value=pad_value
        )
        padded_time_deltas = pad_sequence(
            [s["time_deltas"] for s in x], batch_first=True, padding_value=pad_value
        )
        mask = padded_answers.ne(pad_value)

        return {
            "user_ids": user_ids,
            "skill_ids": skill_ids,
            "item_ids": padded_item_ids,
            "answers": padded_answers,
            "time_deltas": padded_time_deltas,
            "mask": mask,
        }

    @staticmethod
    def collate_fn_stack(x, pad_value: int = -1):
        """Collate function for stacked datasets (history + target)."""
        return tuple(
            KnowledgeTracingDataModule.collate_fn(batch, pad_value=pad_value)
            for batch in list(zip(*x))
        )

    def get_items_to_skills(self) -> torch.Tensor:
        """Generate a validated one-item-to-one-skill mapping (item_id -> skill_id)."""

        # Combine all splits to ensure full coverage.
        dfs = [self.train_df]
        if self.val_df is not None:
            dfs.append(self.val_df)
        if self.test_df is not None:
            dfs.append(self.test_df)

        full_df = pd.concat(dfs, ignore_index=True)
        item_skill_nunique = full_df.groupby(ITEM_COL, sort=False)[SKILL_COL].nunique()
        invalid_items = item_skill_nunique[item_skill_nunique.ne(1)]
        if not invalid_items.empty:
            sample_items = invalid_items.index.tolist()[:10]
            raise ValueError(
                "Each item must map to exactly one skill. "
                f"Found {len(invalid_items)} violating items, e.g. {sample_items}."
            )

        item_skill_map = full_df.groupby(ITEM_COL, sort=False)[SKILL_COL].first()
        missing_items = sorted(set(range(self.stats.num_items)).difference(item_skill_map.index.tolist()))
        if missing_items:
            raise ValueError(
                "Missing item-to-skill mappings for encoded item IDs: "
                f"{missing_items[:10]}{'...' if len(missing_items) > 10 else ''}"
            )

        item_skill_map = item_skill_map.sort_index()
        return torch.as_tensor(item_skill_map.to_numpy().copy(), dtype=torch.long)
