"""Guards against packaging mistakes that only surface for people cloning or installing the package.

An over-broad ``.gitignore`` pattern silently drops source from git. That is invisible in a dev
checkout — the source tree still imports fine — and only breaks on a fresh clone.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "betabern"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_no_package_source_is_gitignored():
    """Every .py file under betabern/ must be visible to git.

    Regression guard: an unanchored ``data/`` pattern once matched the package's own
    ``core/data/`` directory, dropping the whole data layer from the repository.
    """
    if not SRC.exists():  # running against an installed copy, not the repo
        pytest.skip("source tree not present")
    sources = sorted(str(p) for p in SRC.rglob("*.py"))
    assert sources, "no package sources found"

    proc = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(sources),
                          capture_output=True, text=True, cwd=SRC.parent)
    if proc.returncode == 128:
        pytest.skip(f"not a git repository: {proc.stderr.strip()}")
    ignored = [line for line in proc.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these package sources are gitignored and would be missing from a clone and from the "
        "repository:\n  " + "\n  ".join(ignored))


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_expected_subpackages_present():
    """The package's subpackages all exist on disk (catches a partial move/copy)."""
    if not SRC.exists():
        pytest.skip("source tree not present")
    for sub in ("core/data", "core/model", "bernstein/estimator", "irt/estimator", "benchmark", "commands"):
        assert (SRC / sub / "__init__.py").exists(), f"missing subpackage: {sub}"
