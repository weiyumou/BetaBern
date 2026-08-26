"""Lightning training / prediction harness shared by the CLI and programmatic users.

Wraps the repetitive parts of fitting an estimator to a response log: building the datamodule from a
DataFrame, the EarlyStopping + ModelCheckpoint + TensorBoard + Trainer skeleton, and flattening
``predict_step`` output into a per-observation annotated frame (P(correct) + ability posterior
mean/variance, with the original string IDs restored).
"""
import lightning as L
import numpy as np
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from betabern.core.data.datamodule import KnowledgeTracingDataModule
from betabern.core.task import TRAIN_PREFIX


def build_datamodule(df: pd.DataFrame,
                     split: str = "within",
                     train_val_test: tuple[float, float, float] = (0.7, 0.1, 0.2),
                     batch_size: int = 64,
                     num_workers: int = 0,
                     verbose: bool = True) -> KnowledgeTracingDataModule:
    """Build the datamodule from a canonical response-log DataFrame."""
    dm = KnowledgeTracingDataModule(
        df,
        train_val_test_split=tuple(train_val_test),
        split_strategy=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if verbose:
        val_n = len(dm.val_df) if dm.val_df is not None else 0
        test_n = len(dm.test_df) if dm.test_df is not None else 0
        print(f"Data loaded: {dm.stats.num_users} users, {dm.stats.num_skills} skills, "
              f"{dm.stats.num_items} items")
        print(f"Split: {split}  |  Train={len(dm.train_df)}, Val={val_n}, Test={test_n}")
    return dm


def fit(task, dm, *,
        max_epochs: int = 100,
        accelerator: str = "cpu",
        patience: int = 10,
        monitor: str = "1-log-loss/2-val",
        monitor_mode: str = "min",
        log_dir: str = "lightning_logs",
        exp_name: str = "betabern",
        logger: str = "csv",
        test: bool = True) -> L.Trainer:
    """Standard fit -> (test) loop with early stopping, checkpointing, and a metrics logger.

    ``logger`` is ``"csv"`` (default, no extra dependency) or ``"tensorboard"`` (needs the
    ``tensorboard`` package installed).

    When the split leaves no validation data (too few users, or a zero val proportion), the monitored
    metric is never logged, so early stopping is skipped and the checkpoint keeps the last epoch —
    training then simply runs for ``max_epochs``.
    """
    has_val = dm.val_df is not None and len(dm.val_df) > 0
    if not has_val:
        print(f"No validation data after splitting — skipping early stopping on '{monitor}'; "
              f"training for the full {max_epochs} epochs.")
        # The LR scheduler would also condition on the (never-logged) val metric.
        task.monitor = f"1-log-loss/{TRAIN_PREFIX}"
    checkpoint = ModelCheckpoint(monitor=monitor if has_val else None, mode=monitor_mode,
                                 save_top_k=1, filename="best-{epoch}-{step}", verbose=False)
    callbacks = [checkpoint]
    if has_val:
        callbacks.append(EarlyStopping(monitor=monitor, mode=monitor_mode,
                                       patience=patience, verbose=False))
    logger_cls = TensorBoardLogger if logger == "tensorboard" else CSVLogger
    logger = logger_cls(save_dir=log_dir, name=exp_name)
    trainer = L.Trainer(max_epochs=max_epochs, accelerator=accelerator,
                        callbacks=callbacks, logger=logger, log_every_n_steps=2)
    trainer.fit(task, datamodule=dm)
    print(f"\nBest model: {checkpoint.best_model_path}")
    if test and dm.test_df is not None and len(dm.test_df) > 0:
        trainer.test(task, datamodule=dm, ckpt_path="best")
    return trainer


def predict_frame(trainer: L.Trainer, task, dm) -> pd.DataFrame:
    """Predict on the full dataset and return the per-observation annotated frame.

    Columns: ``user_id, skill_name, item_id, is_correct, correct_probability, ability_mean,
    ability_var`` — one row per observed response, IDs decoded back to their original values.
    Filter-style ``(B, S + 1)`` ability trajectories (the online estimator) drop the initial-state
    column so every model annotates the ability *after* observing each response's sequence.
    """
    predictions = trainer.predict(task, datamodule=dm, ckpt_path="best")
    rows = []
    for b in predictions:
        mask = b["mask"]
        lengths = mask.sum(dim=-1).cpu().numpy()
        ability, variance = b["tracked_ability"], b["tracked_variance"]
        if ability.shape[1] == mask.shape[1] + 1:  # (B, S+1) trajectory w/ initial state
            ability, variance = ability[:, 1:], variance[:, 1:]
        rows.append(pd.DataFrame({
            "user_id": np.repeat(b["user_ids"].cpu().numpy(), lengths),
            "skill_name": np.repeat(b["skill_ids"].cpu().numpy(), lengths),
            "item_id": b["item_ids"][mask].cpu().numpy(),
            "is_correct": b["answers"][mask].cpu().numpy(),
            "correct_probability": b["log_p_correct"][mask].exp().cpu().numpy(),
            "ability_mean": ability[mask].cpu().numpy(),
            "ability_var": variance[mask].cpu().numpy(),
        }))
    out = pd.concat(rows, ignore_index=True)
    for col, key in (("user_id", "users"), ("skill_name", "skills"), ("item_id", "items")):
        out[col] = dm.maps[key].inverse_transform(out[col])
    return out
