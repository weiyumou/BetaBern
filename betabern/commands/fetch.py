"""``betabern fetch-irw`` — download Item Response Warehouse tables into the local cache.

Requires the ``irw`` package (not a betabern dependency; heavy redivis + geo stack) and a Redivis
account: set ``REDIVIS_API_TOKEN`` (auto-loaded from a project-root ``.env`` if present). Each table is
normalized to the canonical response-log schema and cached under ``data/irw/<name>.csv``; every other
command then reads the cache with no ``irw`` dependency.

    betabern fetch-irw art spelling_assessment_study1 gilbert_meta_39
"""
import argparse
from pathlib import Path

import pandas as pd

from betabern.core.data import ingest


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("names", nargs="+", help="IRW table names.")
    p.add_argument("--cache-dir", type=Path, default=ingest.IRW_CACHE_DIR,
                   help=f"Cache directory (default: {ingest.IRW_CACHE_DIR}).")
    p.add_argument("--allow-longitudinal", action="store_true",
                   help="Flatten longitudinal (multi-wave) tables keep-first instead of refusing.")


def main(args) -> None:
    for name in args.names:
        out = ingest.fetch_irw(name, cache_dir=args.cache_dir,
                               allow_longitudinal=args.allow_longitudinal)
        d = pd.read_csv(out)
        print(f"{name}: {len(d)} rows, {d['user_id'].nunique()} users, {d['item_id'].nunique()} items "
              f"({d['is_correct'].mean():.3f} correct) -> {out}")
