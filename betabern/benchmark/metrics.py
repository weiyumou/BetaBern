"""
Benchmark metrics for static-ability measurement.

Prediction metrics are computed from a flat ``(p_correct, y)`` pair (held-out responses, any model);
recovery metrics compare to the simulator's ground truth and are scale-free so they're comparable
across model families (IRT lives on ``theta in R``, Bernstein on ``[0, 1]``).
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

EPS = 1e-7


def prediction_metrics(p_correct: np.ndarray, y: np.ndarray) -> dict:
    """Held-out prediction quality from predicted P(correct) and binary outcomes.

    :param p_correct: predicted P(correct), shape ``(N,)`` in ``[0, 1]``
    :param y: observed outcomes, shape ``(N,)`` in ``{0, 1}``
    :return: ``{n, nll, auc, brier, acc, ece}`` (``auc`` is ``nan`` if ``y`` has a single class)
    """
    p = np.clip(np.asarray(p_correct, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)

    p_obs = np.where(y > 0.5, p, 1 - p)               # P(observed class)
    nll = float(-np.log(p_obs).mean())
    brier = float(np.mean((p - y) ** 2))
    acc = float(np.mean((p >= 0.5) == (y > 0.5)))
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return {"n": int(y.size), "nll": nll, "auc": auc, "brier": brier, "acc": acc,
            "ece": expected_calibration_error(p, y)}


def expected_calibration_error(p_correct: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width-binned |confidence - accuracy|, weighted by bin population."""
    p = np.asarray(p_correct, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def theta_recovery(theta_hat: np.ndarray, theta_true: np.ndarray) -> dict:
    """Per-(user, skill) ability recovery. Rank-based (Spearman) is the headline — it's invariant to
    the monotone reparameterization between IRT's and Bernstein's latent scales; Pearson is reported too.
    """
    theta_hat, theta_true = np.asarray(theta_hat, dtype=np.float64), np.asarray(theta_true, dtype=np.float64)
    if theta_hat.size < 2 or np.std(theta_hat) < EPS or np.std(theta_true) < EPS:
        return {"theta_spearman": float("nan"), "theta_pearson": float("nan")}
    return {"theta_spearman": float(spearmanr(theta_hat, theta_true).statistic),
            "theta_pearson": float(pearsonr(theta_hat, theta_true).statistic)}


def prob_recovery(p_model: np.ndarray, p_true: np.ndarray) -> dict:
    """Per-interaction recovery of the generative P(correct) (both in ``[0, 1]`` -> directly comparable)."""
    p_model, p_true = np.asarray(p_model, dtype=np.float64), np.asarray(p_true, dtype=np.float64)
    mae = float(np.mean(np.abs(p_model - p_true)))
    if p_model.size < 2 or np.std(p_model) < EPS or np.std(p_true) < EPS:
        return {"prob_mae": mae, "prob_pearson": float("nan")}
    return {"prob_mae": mae, "prob_pearson": float(pearsonr(p_model, p_true).statistic)}
