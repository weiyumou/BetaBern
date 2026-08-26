"""Dataset-parallel benchmark execution.

``runner.run_benchmark`` sweeps datasets serially. For the multi-dataset paper run each dataset is
independent (its own folds, fits, and scoring), so this runs them in **separate processes** — CPU-bound
torch work that the GIL would otherwise serialize — and recombines the per-dataset frames into exactly the
``(runs, summary, predictions)`` the serial runner returns. Threads per worker are capped so ``W`` processes
don't oversubscribe the cores (which would thrash and run *slower* than serial).
"""
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from betabern.benchmark.runner import _summarize, run_benchmark


def _run_one_dataset(payload: tuple[dict, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Worker entry: run the full model sweep for a single-dataset sub-config in this process."""
    sub_config, threads = payload
    import torch
    torch.set_num_threads(max(1, threads))  # bound intra-op parallelism so W workers share the cores
    warnings.filterwarnings("ignore")
    for name in ("lightning.pytorch", "lightning.pytorch.utilities.rank_zero", "pytorch_lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    runs, _summary, predictions = run_benchmark(sub_config, verbose=True)
    return runs, predictions


def run_benchmark_parallel(config: dict, max_workers: int | None = None,
                           verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run each dataset in its own process, then recombine — drop-in for :func:`run_benchmark`.

    Returns the same ``(runs, summary, predictions)``; ``summary`` and the paired tests (computed by the
    caller) are recomputed from the combined ``runs``, so the result is identical to a serial run, just
    faster. Falls back to the serial runner when there is nothing to parallelize (<= 1 dataset).
    """
    datasets = config["datasets"]
    if len(datasets) <= 1:
        return run_benchmark(config, verbose=verbose)

    cores = os.cpu_count() or len(datasets)
    n_workers = min(len(datasets), max_workers or len(datasets), cores)
    threads = max(1, cores // n_workers)
    subs = [{**config, "datasets": [dspec]} for dspec in datasets]  # one independent run per dataset

    runs_frames, pred_frames = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_one_dataset, (sub, threads)) for sub in subs]
        for fut in as_completed(futures):
            runs, predictions = fut.result()
            runs_frames.append(runs)
            pred_frames.append(predictions)

    runs = pd.concat(runs_frames, ignore_index=True)
    predictions = (pd.concat(pred_frames, ignore_index=True)
                   if any(len(p) for p in pred_frames) else pd.DataFrame())
    return runs, _summarize(runs), predictions
