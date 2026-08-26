"""Data ingestion: response-log tables and Item Response Warehouse (IRW) tables.

The package operates on one canonical response-log schema::

    user_id | skill_name | item_id | is_correct | timestamp

Two input forms are accepted and mapped onto it by :func:`load_table`:

- **Response-log CSVs** (the schema above; ``timestamp`` is synthesized when absent, since static
  estimation is order-invariant).
- **IRW-shaped tables** (``id | item | resp [, rt, wave, date, ...]`` — the long format served by the
  ``irw`` package), normalized by :func:`normalize_irw`.

Downloading from IRW itself (:func:`fetch_irw`) is the only step that needs the ``irw`` package and a
Redivis token; it is imported lazily so the runtime never depends on it. Fetched tables are cached under
``data/irw/`` and thereafter read like any other CSV.
"""
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

IRW_CACHE_DIR = Path("data/irw")
LOG_COLS = ("user_id", "skill_name", "item_id", "is_correct")
IRW_COLS = ("id", "item", "resp")
LONGITUDINAL_COLS = ("wave", "date")  # IRW's longitudinal markers (repeated measurement occasions)


def synthetic_timestamps(n: int) -> pd.Series:
    """A monotone stand-in ``timestamp`` column (static estimation is order-invariant)."""
    return pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(n), unit="s")


def normalize_irw(raw: pd.DataFrame, name: str, allow_longitudinal: bool = False) -> pd.DataFrame:
    """Map a raw IRW long-format table (``id, item, resp``) onto the canonical response-log schema.

    Dichotomous responses only; one response per (person, item) (multi-wave / ``density>1`` tables
    collapse keep-first); the single construct becomes one ``skill_name`` (unidimensional).

    Refuses *longitudinal* tables (a ``wave``/``date`` column = repeated measurement occasions) by
    default: static-trait estimation assumes a single occasion, and the keep-first flatten would
    silently muddy multi-wave data. Pass ``allow_longitudinal=True`` to flatten anyway.
    """
    longi = [c for c in raw.columns if c.lower() in LONGITUDINAL_COLS]
    if longi and not allow_longitudinal:
        raise ValueError(
            f"IRW table '{name}' is longitudinal (has {longi} -> repeated measurement occasions); "
            f"static-trait estimation assumes one occasion. Pick a non-longitudinal table, or pass "
            f"allow_longitudinal=True (--allow-longitudinal) to flatten it (keep-first).")
    df = (raw[["id", "item", "resp"]]
          .rename(columns={"id": "user_id", "item": "item_id", "resp": "is_correct"}))
    df = df[df["is_correct"].isin([0, 1])].copy()  # dichotomous only
    df["is_correct"] = df["is_correct"].astype(int)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")  # one response per cell
    df["skill_name"] = name  # single construct -> unidimensional
    df["timestamp"] = synthetic_timestamps(len(df))
    return df[["user_id", "skill_name", "item_id", "is_correct", "timestamp"]]


def load_table(path: str | Path, skills: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Load a CSV in either accepted form and return the canonical response log.

    Auto-detects the form from the header: the canonical ``user_id/skill_name/item_id/is_correct``
    columns are taken as a response log (``timestamp`` synthesized when absent); ``id/item/resp``
    columns are taken as an IRW-shaped table and normalized (``skill_name`` = file stem).
    """
    path = Path(path)
    header = set(pd.read_csv(path, nrows=0).columns)
    if header.issuperset(LOG_COLS):
        df = pd.read_csv(path, parse_dates=["timestamp"] if "timestamp" in header else False)
        if "timestamp" not in header:
            df["timestamp"] = synthetic_timestamps(len(df))
    elif header.issuperset(IRW_COLS):
        df = normalize_irw(pd.read_csv(path), name=path.stem)
    else:
        raise ValueError(
            f"{path}: unrecognized columns {sorted(header)}. Expected a response log "
            f"{list(LOG_COLS)} (+ optional timestamp) or an IRW-shaped table {list(IRW_COLS)}.")
    df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
    if skills is not None:
        df = df[df["skill_name"].isin(skills)].copy()
    return df


# ======================================================================
# IRW download (the only part that needs the `irw` package + a Redivis token)
# ======================================================================

def load_env_token() -> None:
    """Populate ``REDIVIS_API_TOKEN`` from a project-root ``.env`` if not already in the environment."""
    if os.environ.get("REDIVIS_API_TOKEN"):
        return
    for base in (Path.cwd(), *Path.cwd().parents):
        env = base / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                key, _, val = line.strip().partition("=")
                if key == "REDIVIS_API_TOKEN" and val:
                    os.environ["REDIVIS_API_TOKEN"] = val.strip().strip('"').strip("'")
            return


def fetch_irw(name: str, cache_dir: Path = IRW_CACHE_DIR, allow_longitudinal: bool = False) -> Path:
    """Download IRW table ``name``, normalize it, and cache to ``<cache_dir>/<name>.csv``.

    Requires the ``irw`` package (heavy: redivis + geo stack) and a Redivis account
    (``REDIVIS_API_TOKEN``, auto-loaded from a project-root ``.env`` if present).
    """
    try:
        import irw  # deliberately not a package dependency
    except ImportError as exc:
        raise ImportError(
            "Downloading IRW tables needs the 'irw' package, which is not a betabern dependency "
            "(heavy transitive stack). Install it (e.g. `uv pip install irw`) or fetch in a dedicated "
            "env, then re-run — cached tables are read without it.") from exc

    load_env_token()
    df = normalize_irw(irw.fetch(name), name, allow_longitudinal=allow_longitudinal)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{name}.csv"
    df.to_csv(out, index=False)
    return out


def resolve_irw(name: str, cache_dir: Path = IRW_CACHE_DIR, fetch: bool = True) -> Path:
    """Return the cached CSV path for IRW table ``name``, downloading it first when allowed & needed."""
    path = cache_dir / f"{name}.csv"
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(
                f"IRW table '{name}' is not cached at {path}. Fetch it first: betabern fetch-irw {name}")
        fetch_irw(name, cache_dir=cache_dir)
    return path
