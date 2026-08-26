"""Measurement-science read-outs for static-ability estimators.

Two analyses that consume a **trained** :class:`~betabern.core.model.estimator.QuadratureEstimator`
(no extra fitting), supporting the paper's "what the closed form buys" claims:

- :func:`measurement_curve` — sequential measurement efficiency: as a student answers more (pre-calibrated)
  items, how do posterior uncertainty and one-step predictive accuracy evolve? (Uses :meth:`prequential`;
  this is online *measurement*, not knowledge tracing — the trait is static, no learning transition.)
- :func:`coverage_stats` — posterior-uncertainty calibration on simulated data with known ability:
  θ-recovery + whether the closed-form credible interval has nominal coverage.

Everything is reported on the simulator's ability scale ``R``; a ``[0, 1]`` (Beta-Bernstein) model's
ability is mapped through the bridge link ``eta = 6θ - 3`` so families are comparable.
"""
import numpy as np
import pandas as pd
import torch

from betabern.core.model.prior import BetaPrior

ETA_LINK = (-3.0, 3.0)  # bridge link: theta in [0, 1] <-> eta in [-3, 3] (eta = 6θ - 3)


def _to_eta(model):
    """Map a model's ability read-out to the simulator's ``R`` scale: identity for a Normal-prior (ℝ)
    model, the affine link ``6θ - 3`` for a Beta-prior (``[0, 1]``) one."""
    if isinstance(model.prior, BetaPrior):
        lo, hi = ETA_LINK
        return lambda x: lo + (hi - lo) * x
    return lambda x: x


def _eta_scale(model) -> float:
    """Linear factor taking a model-scale **SD** to the ``R`` (η) scale: the link's slope ``hi - lo`` for a
    ``[0, 1]`` Beta model, 1 for a Normal (ℝ) model. (An SD scales by the affine map's slope only.)"""
    if isinstance(model.prior, BetaPrior):
        lo, hi = ETA_LINK
        return hi - lo
    return 1.0


@torch.no_grad()
def measurement_curve(model, skill_ids: torch.Tensor, item_ids: torch.Tensor, answers: torch.Tensor,
                      n_orders: int = 20, seed: int = 0) -> pd.DataFrame:
    """Sequential measurement-efficiency curve, averaged over ``n_orders`` random item orders.

    Inputs are **complete** response matrices ``(B, S)`` — one row per student, all ``S`` items answered
    (IRW responses have no intrinsic order, so we average over random administration orders). Requires a
    model with :meth:`prequential` (a ``QuadratureEstimator``).

    :return: tidy DataFrame, one row per prefix length ``k``:
        - ``post_sd``  — mean posterior SD (on the ℝ/η scale, so families are comparable) after ``k`` items
          observed (``k = 0..S``; ``k = 0`` is the prior),
        - ``preq_nll`` — mean ``-log p(y | k prior items)`` for the one-step prediction of item ``k+1``
          (``k = 0..S-1``; ``NaN`` at ``k = S``),
        - ``preq_acc`` — mean accuracy of that one-step prediction.
    """
    B, S = item_ids.shape
    mask = torch.ones(B, S, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    sd = torch.zeros(S + 1)
    nll = torch.zeros(S)
    acc = torch.zeros(S)
    for _ in range(n_orders):
        perm = torch.argsort(torch.rand(B, S, generator=g), dim=1)  # per-student random order
        it, an = item_ids.gather(1, perm), answers.gather(1, perm)
        pred, _mean, std = model.prequential(skill_ids, it, an, mask)  # (B,S,2), (B,S+1), (B,S+1)
        sd += std.mean(dim=0)
        obs = pred.gather(2, an[:, :, None]).squeeze(-1)  # (B, S) prequential log p(y_k | y_<k)
        nll += (-obs).mean(dim=0)
        acc += ((pred[:, :, 1] > pred[:, :, 0]) == an.bool()).float().mean(dim=0)
    sd = sd / n_orders * _eta_scale(model)  # report posterior SD on the ℝ (η) scale -> comparable across families
    nll, acc = nll / n_orders, acc / n_orders
    return pd.DataFrame({
        "k_items": np.arange(S + 1),
        "post_sd": sd.numpy(),
        "preq_nll": np.append(nll.numpy(), np.nan),
        "preq_acc": np.append(acc.numpy(), np.nan),
    })


@torch.no_grad()
def coverage_stats(model, skill_ids: torch.Tensor, item_ids: torch.Tensor, answers: torch.Tensor,
                   mask: torch.Tensor, true_theta: torch.Tensor,
                   levels: tuple[float, ...] = (0.5, 0.8, 0.95)) -> dict:
    """Posterior-uncertainty calibration on simulated data with known ability ``true_theta`` (on ``R``).

    Reports θ-recovery (Pearson correlation and RMSE of the EAP) and **credible-interval coverage**: does
    the model's closed-form ``x%`` interval contain the true θ for ~``x%`` of students? A ``[0, 1]`` model's
    EAP/CI are mapped to ``R`` via the link before comparison.

    :return: ``{"theta_corr", "theta_rmse", "cov_50", "cov_80", "cov_95", ...}``.
    """
    to_eta = _to_eta(model)
    eap = to_eta(model.posterior_stats(skill_ids, item_ids, answers, mask, level=None).mean)
    out = {"theta_corr": float(torch.corrcoef(torch.stack([eap, true_theta]))[0, 1]),
           "theta_rmse": float(((eap - true_theta) ** 2).mean().sqrt())}
    for lvl in levels:
        p = model.posterior_stats(skill_ids, item_ids, answers, mask, level=lvl)
        lo_e, hi_e = to_eta(p.ci_low), to_eta(p.ci_high)
        out[f"cov_{int(round(lvl * 100))}"] = float(((true_theta >= lo_e) & (true_theta <= hi_e)).float().mean())
    return out
