"""
Resolve a benchmark dataset spec to a DataFrame + a ground-truth flag.

A spec is one of: ``{"path": "...", "skills": [...]}`` (an arbitrary response-log CSV),
``{"irw": "<table>"}`` (an Item Response Warehouse table, fetched + cached to ``data/irw/``), or
``{"simulate": {...}}`` (regenerate static IRT data in-memory, with full ground truth). Ground truth
(``user_theta``, ``p_true_correct``) enables the recovery metrics; real CSVs / IRW tables get prediction
metrics only, unless they happen to carry those columns.
"""
import pandas as pd

from betabern.core.data.datamodule import load_data
from betabern.core.data.ingest import IRW_CACHE_DIR, resolve_irw
from betabern.core.data.simulate_irt import generate_log_data

TRUTH_COLS = ("user_theta", "p_true_correct")


def resolve(spec: dict) -> tuple[pd.DataFrame, bool]:
    """Return ``(df, has_truth)`` for a dataset spec (``{"path": ...}``, ``{"irw": ...}``, or ``{"simulate": {...}}``)."""
    if "path" in spec:
        df = load_data(spec["path"], skills=spec.get("skills"))
    elif "irw" in spec:
        # IRW tables are fetched out-of-band by `betabern fetch-irw` (needs the `irw` package + a
        # Redivis token) and cached under data/irw/. We only read the cache here (fetch=False), so the
        # benchmark never depends on `irw`.
        path = resolve_irw(spec["irw"], cache_dir=IRW_CACHE_DIR, fetch=False)
        df = load_data(str(path), skills=spec.get("skills"))
    elif "simulate" in spec:
        params = dict(spec["simulate"])
        params.setdefault("drift_scale", 0.0)  # static ability by default (the measurement setting)
        log_df, _item_df = generate_log_data(**params)
        df = log_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    else:
        raise ValueError(f"dataset spec needs a 'path', 'irw', or 'simulate' key, got {list(spec)}")

    has_truth = all(c in df.columns for c in TRUTH_COLS)
    return df, has_truth
