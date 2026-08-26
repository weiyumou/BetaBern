"""
Benchmark sweep: fit each model on each dataset/split/fold, run every model's ``predict_step`` to
materialize one tidy row per scored interaction (predicted P(correct) + estimated ability), and compute
all statistics from that frame. Returns per-run + aggregated DataFrames plus the row-level predictions.
"""
import sys
import tempfile

import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from betabern.benchmark import datasets, metrics, registry
from betabern.core.data.datamodule import (
    ITEM_COL,
    SKILL_COL,
    USER_COL,
    KnowledgeTracingDataModule,
)
from betabern.core.data.splitter import DataSplitter
from betabern.core.util import count_free_params

_MONITOR = "1-log-loss/2-val"
_DEFAULT_TRAINER = {"max_epochs": 200, "patience": 10, "min_delta": 1e-4, "accelerator": "cpu",
                    "batch_size": 64, "train_val_test": (0.7, 0.1, 0.2)}


def _fit(task, dm, tcfg, ckpt_dir: str):
    """Fit with early stopping + best-model checkpointing; return the trainer.

    The checkpoint (best validation log-loss, saved under the temp ``ckpt_dir``) lets prediction and
    recovery use each model's *best* weights — via ``ckpt_path="best"`` — rather than the last
    (overshoot) epoch, which would unfairly penalize fast-converging models.
    """
    callbacks = []
    if dm.val_df is not None:
        callbacks = [ModelCheckpoint(dirpath=ckpt_dir, monitor=_MONITOR, mode="min", save_top_k=1),
                     EarlyStopping(monitor=_MONITOR, mode="min",
                                   patience=tcfg["patience"], min_delta=tcfg["min_delta"])]
    trainer = L.Trainer(max_epochs=tcfg["max_epochs"], accelerator=tcfg["accelerator"],
                        callbacks=callbacks, logger=False,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(task, datamodule=dm)
    return trainer


def predictions_to_frame(preds: list[dict]) -> pd.DataFrame:
    """Flatten per-batch ``predict_step`` outputs into one tidy row per scored (valid) interaction.

    Depends only on the shared predict_step contract (``log_p_correct``, ``answers``, ``item_ids``,
    ``tracked_ability``, ``mask``, ``user_ids``, ``skill_ids``), so the same flattener serves every
    model and both protocols. Columns: ``user_id, skill_name, item_id, y, p_correct, ability`` (ids in
    the datamodule's encoded space) — the single source every benchmark statistic is computed from.
    """
    frames = []
    for b in preds:
        mask = b["mask"]
        n = mask.sum(dim=-1).cpu().numpy()               # valid responses per (user, skill) row
        ability = b["tracked_ability"]
        if ability.shape[1] == mask.shape[1] + 1:        # filter-style (B, S + 1) trajectory
            ability = ability[:, :-1]
        frames.append(pd.DataFrame({
            # per-(user, skill) scalars broadcast to their interactions; (B, S) tensors masked row-major
            USER_COL: np.repeat(b["user_ids"].cpu().numpy(), n),
            SKILL_COL: np.repeat(b["skill_ids"].cpu().numpy(), n),
            ITEM_COL: b["item_ids"][mask].cpu().numpy(),
            "y": b["answers"][mask].cpu().numpy(),
            "p_correct": b["log_p_correct"][mask].exp().cpu().numpy(),
            "ability": ability[mask].cpu().numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def _attach_truth(frame: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join the simulator's ground truth onto a recovery frame (ids are in the datamodule's encoded
    space, shared with ``df``). For a static trait (``drift_scale=0``) the truth is a pure function of
    identifiers: ``user_theta`` per (user, skill), ``p_true_correct`` per (user, item)."""
    theta = df.drop_duplicates([USER_COL, SKILL_COL])[[USER_COL, SKILL_COL, "user_theta"]]
    ptrue = df.drop_duplicates([USER_COL, ITEM_COL])[[USER_COL, ITEM_COL, "p_true_correct"]]
    return (frame.merge(theta, on=[USER_COL, SKILL_COL], how="left")
            .merge(ptrue, on=[USER_COL, ITEM_COL], how="left"))


def _decode_ids(frame: pd.DataFrame, dm) -> pd.DataFrame:
    """Map the datamodule's encoded indices back to the dataset's original identifiers (done after the
    encoded-space truth join), so the saved predictions are joinable to the source data."""
    for col, key in ((USER_COL, "users"), (SKILL_COL, "skills"), (ITEM_COL, "items")):
        frame[col] = dm.maps[key].inverse_transform(frame[col].to_numpy())
    return frame


def _count_params(model) -> int:
    """Free-parameter capacity, reported alongside every run so wins/losses can be read against the
    budget (e.g. static_irt's incidental θ, or a free-Bernstein capacity edge). Uses the
    ``num_free_params`` protocol so softmax-simplex models aren't over-counted (see ``core.util``)."""
    return count_free_params(model)


def _prediction_run(df, key, split, fold, train_inds, test_inds, hp, tcfg) -> tuple[pd.DataFrame, int]:
    """Fit on this fold's train, score its held-out (stacked) test set with the best weights.

    ``train_inds``/``test_inds`` are the fold's partition (shared across models so comparisons are
    paired); the datamodule carves an internal validation slice from the train portion for early stopping.
    Returns the row-level predictions frame and the model's parameter count.
    """
    L.seed_everything(fold, workers=True)
    dm = KnowledgeTracingDataModule(df, train_inds=train_inds, test_inds=test_inds,
                                    train_val_test_split=tcfg["train_val_test"],
                                    split_strategy=split, batch_size=tcfg["batch_size"])
    model, task = registry.build(key, dm, hp)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        trainer = _fit(task, dm, tcfg, ckpt_dir)
        dm.setup("test")
        ckpt_path = "best" if dm.val_df is not None else None
        preds = trainer.predict(task, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_path)
    return _decode_ids(predictions_to_frame(preds), dm), _count_params(model)


def _recovery_run(df, key, hp, tcfg) -> tuple[pd.DataFrame, int]:
    """Fit once on the full data (items calibrated), then score every (user, skill) from its own complete
    response vector via the task's ``predict_step`` on self-stacked ``(full, full)`` batches — identical
    to estimating θ and posterior-predictive P(correct) for the pair. Ground truth is joined on for the
    recovery metrics. Run once per dataset×model (no held-out fold), not cross-validated.
    """
    L.seed_everything(0, workers=True)
    dm = KnowledgeTracingDataModule(df, train_val_test_split=(0.9, 0.1, 0.0),
                                    split_strategy="within_random", batch_size=tcfg["batch_size"])
    model, task = registry.build(key, dm, hp)
    with tempfile.TemporaryDirectory() as ckpt_dir:
        trainer = _fit(task, dm, tcfg, ckpt_dir)
        ckpt_path = "best" if dm.val_df is not None else None  # restore best weights for scoring
        preds = trainer.predict(task, dataloaders=dm.recovery_dataloader(), ckpt_path=ckpt_path)
    frame = _attach_truth(predictions_to_frame(preds), dm.df)  # join truth in encoded space, then decode
    return _decode_ids(frame, dm), _count_params(model)


def _score(frame: pd.DataFrame, split: str) -> dict:
    """All statistics for one run, computed from its predictions frame. Prediction splits score
    held-out (p_correct, y); recovery scores θ (deduped to one estimate per (user, skill)) and the
    generative P(correct) per interaction against the joined ground truth."""
    if split == "recovery":
        per_pair = frame.drop_duplicates([USER_COL, SKILL_COL])
        return {**metrics.theta_recovery(per_pair["ability"].to_numpy(), per_pair["user_theta"].to_numpy()),
                **metrics.prob_recovery(frame["p_correct"].to_numpy(), frame["p_true_correct"].to_numpy())}
    return metrics.prediction_metrics(frame["p_correct"].to_numpy(), frame["y"].to_numpy())


def _resolve_model(entry, base_hp) -> tuple[str, str, dict]:
    """A models entry is a registry key (str) or ``{"key", "name"?, "overrides"?}`` -> (key, label, hp)."""
    if isinstance(entry, str):
        return entry, entry, base_hp
    key = entry["key"]
    return key, entry.get("name", key), {**base_hp, **entry.get("overrides", {})}


def _log_progress(k: int, total: int, dataset: str, model: str, split: str, fold: int, row: dict):
    """One-line progress to stderr (stdout stays reserved for the final summary table)."""
    keys = ("theta_spearman", "prob_mae") if split == "recovery" else ("nll", "auc")
    bits = "  ".join(f"{key}={row[key]:.3f}" for key in keys if row.get(key) == row.get(key))  # skip NaN
    print(f"[{k}/{total}] {dataset} | {model} | {split} | fold={fold}  {bits}", file=sys.stderr, flush=True)


def run_benchmark(config: dict, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full sweep. Returns ``(runs, summary, predictions)`` DataFrames.

    Prediction protocols use K-fold cross-validation (``n_folds``, default 5): folds are generated once
    per (dataset, split) and shared across models, so comparisons are paired. ``predictions`` has one row
    per scored interaction, tagged with ``(dataset, model, split, fold)`` — the row-level artifact every
    statistic is computed from. ``runs`` aggregates those rows to one metric row per (dataset, model,
    split, fold); recovery rows (fit once per dataset×model on the full data, not cross-validated) use
    ``split="recovery"``. ``summary`` aggregates ``runs`` mean±std over folds per (dataset, model, split).
    With ``verbose``, prints per-run progress to stderr.
    """
    base_hp = {**registry.DEFAULT_HP, **config.get("hyperparams", {})}
    tcfg = {**_DEFAULT_TRAINER, **config.get("trainer", {})}
    n_folds = config.get("n_folds", 5)

    resolved = [(dspec.get("name") or dspec.get("path", "dataset"), *datasets.resolve(dspec))
                for dspec in config["datasets"]]
    models = [_resolve_model(entry, base_hp) for entry in config["models"]]
    total = (len(resolved) * len(models) * len(config["splits"]) * n_folds
             + sum(has_truth for _, _, has_truth in resolved) * len(models))

    rows, frames, k = [], [], 0

    def _record(frame, n_params, name, label, split, fold):
        nonlocal k
        row = _score(frame, split)
        rows.append({"dataset": name, "model": label, "split": split, "fold": fold,
                     "n_params": n_params, **row})
        frames.append(frame.assign(dataset=name, model=label, split=split, fold=fold))
        k += 1
        if verbose:
            _log_progress(k, total, name, label, split, fold, row)

    for name, df, has_truth in resolved:
        # Folds are built once per (dataset, split) and reused across models, so every model is scored
        # on identical partitions -> per-fold paired comparisons are valid.
        folds = {split: list(DataSplitter(strategy=split, n_splits=n_folds, random_state=0).split(df))
                 for split in config["splits"]}
        for key, label, hp in models:
            for split in config["splits"]:
                for fold, (train_inds, test_inds) in enumerate(folds[split]):
                    frame, n_params = _prediction_run(df, key, split, fold, train_inds, test_inds, hp, tcfg)
                    _record(frame, n_params, name, label, split, fold)
            if has_truth:
                frame, n_params = _recovery_run(df, key, hp, tcfg)
                _record(frame, n_params, name, label, "recovery", 0)

    runs = pd.DataFrame(rows)
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return runs, _summarize(runs), predictions


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over folds, grouped by (dataset, model, split); ``n_params`` carried through as-is."""
    if runs.empty:
        return runs
    id_cols = ["dataset", "model", "split"]
    metric_cols = [c for c in runs.columns if c not in (*id_cols, "fold", "n", "n_params")]
    agg = runs.groupby(id_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    if "n_params" in runs.columns:  # capacity is constant over folds -> report the single value
        agg = agg.merge(runs.groupby(id_cols)["n_params"].first().reset_index(), on=id_cols)
        agg = agg[[*id_cols, "n_params", *[c for c in agg.columns if c not in (*id_cols, "n_params")]]]
    return agg.dropna(axis=1, how="all")
