"""Shared test fixtures.

The smoke tests read ``data/simulated_student_log.csv``, which is not shipped with the
repository (``data/`` is gitignored). Regenerate it deterministically when absent so the
suite passes from a fresh clone.
"""
from pathlib import Path

import pytest

_SIM_LOG = Path("data/simulated_student_log.csv")


@pytest.fixture(scope="session", autouse=True)
def simulated_student_log():
    if not _SIM_LOG.exists():
        from betabern.core.data.simulate_irt import generate_log_data

        _SIM_LOG.parent.mkdir(exist_ok=True)
        log_df, _ = generate_log_data(100, 20, 80, 100, n_skills=5,
                                      drift_scale=0.05, random_state=42)
        # Match the shape of the original on-disk log: p_true_correct kept, user_theta dropped
        # (test_resolve_csv_without_full_truth relies on the absent user_theta column).
        log_df.drop(columns=["user_theta"]).to_csv(_SIM_LOG, index=False)
    return _SIM_LOG
