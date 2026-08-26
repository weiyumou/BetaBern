"""End-to-end tests for the `betabern` CLI and the data-ingestion layer."""
import pandas as pd
import pytest

from betabern.cli import main
from betabern.core.data import ingest

# ======================================================================
# Ingestion
# ======================================================================

def test_normalize_irw_maps_to_canonical_schema():
    raw = pd.DataFrame({
        "id": [1, 1, 2, 2, 2, 3],
        "item": ["a", "b", "a", "a", "b", "a"],   # user 2 answers item a twice -> keep first
        "resp": [1, 0, 0, 1, 2, 1],               # resp=2 (polytomous) row dropped
    })
    df = ingest.normalize_irw(raw, "mytable")
    assert list(df.columns) == ["user_id", "skill_name", "item_id", "is_correct", "timestamp"]
    assert (df["skill_name"] == "mytable").all()
    assert len(df) == 4  # 6 rows - 1 duplicate cell - 1 non-dichotomous
    assert df.set_index(["user_id", "item_id"])["is_correct"].to_dict() == {
        (1, "a"): 1, (1, "b"): 0, (2, "a"): 0, (3, "a"): 1}


def test_normalize_irw_refuses_longitudinal():
    raw = pd.DataFrame({"id": [1], "item": ["a"], "resp": [1], "wave": [1]})
    with pytest.raises(ValueError, match="longitudinal"):
        ingest.normalize_irw(raw, "multi")
    assert len(ingest.normalize_irw(raw, "multi", allow_longitudinal=True)) == 1


def test_load_table_autodetects_forms(tmp_path):
    # IRW-shaped CSV -> normalized, skill from file stem
    irw_csv = tmp_path / "sometable.csv"
    pd.DataFrame({"id": [1, 2], "item": ["x", "y"], "resp": [0, 1]}).to_csv(irw_csv, index=False)
    df = ingest.load_table(irw_csv)
    assert (df["skill_name"] == "sometable").all()

    # response log without timestamp -> synthesized
    log_csv = tmp_path / "log.csv"
    pd.DataFrame({"user_id": [1, 1], "skill_name": ["s", "s"], "item_id": [1, 2],
                  "is_correct": [0, 1]}).to_csv(log_csv, index=False)
    df = ingest.load_table(log_csv)
    assert df["timestamp"].is_monotonic_increasing

    # unrecognized columns -> clear error
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="unrecognized columns"):
        ingest.load_table(bad_csv)


# ======================================================================
# fit / simulate end-to-end
# ======================================================================

@pytest.fixture(scope="module")
def tiny_log(tmp_path_factory):
    """A tiny simulated log written through the CLI's own `simulate` subcommand."""
    out = tmp_path_factory.mktemp("cli") / "tiny.csv"
    main(["simulate", "--out", str(out), "--num-students", "12", "--num-items", "6",
          "--num-skills", "1", "--min-interactions", "12", "--max-interactions", "16",
          "--seed", "7"])
    return out


@pytest.mark.parametrize("model,extra", [
    ("bern-exact", ["--degree", "4"]),
    ("bern-online", ["--degree", "4", "--weights", "free", "--num-nodes", "8"]),
    ("irt-mml", ["--num-nodes", "8"]),
])
def test_fit_writes_annotated_predictions(tiny_log, tmp_path, model, extra):
    run_dir = tmp_path / "run"
    main(["fit", "--data", str(tiny_log), "--model", model, *extra,
          "--max-epochs", "2", "--batch-size", "8", "--out", str(run_dir),
          "--log-dir", str(tmp_path / "logs"), "--no-test"])
    pred = pd.read_csv(run_dir / "predictions.csv")
    assert {"user_id", "skill_name", "item_id", "is_correct", "correct_probability",
            "ability_mean", "ability_var"} <= set(pred.columns)
    src = pd.read_csv(tiny_log)
    assert len(pred) == len(src)  # one annotated row per observed response
    assert pred["correct_probability"].between(0, 1).all()
    assert (pred["ability_var"] >= 0).all()
    assert (run_dir / "args.json").exists()
    # decoded IDs round-trip back to the source values
    assert set(pred["user_id"]) <= set(src["user_id"])
    assert set(pred["item_id"]) <= set(src["item_id"])


def test_fit_without_validation_split(tmp_path):
    """A dataset too small to yield a validation split trains instead of crashing.

    Both early stopping and the LR scheduler condition on the val log-loss, which is never logged
    when the split leaves no validation rows.
    """
    log = tmp_path / "tiny.csv"
    main(["simulate", "--out", str(log), "--num-students", "8", "--num-items", "4",
          "--num-skills", "1", "--min-interactions", "6", "--max-interactions", "8"])
    run_dir = tmp_path / "run"
    main(["fit", "--data", str(log), "--model", "bern-exact", "--degree", "4",
          "--max-epochs", "2", "--batch-size", "8", "--out", str(run_dir),
          "--log-dir", str(tmp_path / "logs"), "--no-test"])
    pred = pd.read_csv(run_dir / "predictions.csv")
    assert len(pred) == len(pd.read_csv(log))


# ======================================================================
# Top-level dispatch (lazy per-command imports)
# ======================================================================

def test_help_lists_all_commands(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    for command in ("fit", "simulate", "fetch-irw", "benchmark", "report", "merge"):
        assert command in out


def test_no_args_prints_help(capsys):
    main([])
    assert "usage: betabern" in capsys.readouterr().out


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "betabern" in capsys.readouterr().out


def test_unknown_command_exits_with_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["bogus"])
    assert excinfo.value.code == 2
    assert "invalid command" in capsys.readouterr().err


def test_subcommand_help_via_lazy_dispatch(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["report", "--help"])
    assert excinfo.value.code == 0
    assert "run_dir" in capsys.readouterr().out
