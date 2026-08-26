"""``betabern report`` — re-render the Markdown report for a benchmark result dir.

    betabern report results/benchmark/<run>            # to stdout
    betabern report results/benchmark/<run> --out report.md
"""
import argparse
from pathlib import Path

from betabern.benchmark.report import render_markdown


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir", help="Benchmark result directory.")
    p.add_argument("--out", default=None, help="Optional Markdown output path (default: stdout).")


def main(args) -> None:
    report = render_markdown(args.run_dir)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
    print(report, end="")
