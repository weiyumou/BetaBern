"""
Fixed ability priors and their quadrature rules.

A prior here is just a pair of fixed scalar parameters — the latent metric is anchored, nothing is
learned — optionally bundled with the numerical quadrature that integrates against it,
``E_prior[f] = sum_q v_q f(theta_q)``. Keeping the prior and its nodes together lets every static
model share one definition: a :class:`~betabern.core.estimator.QuadratureEstimator` reads
``get_params``/``quadrature`` off its prior, the exact estimator reads only ``get_params`` (it has no
nodes), and the online filter reuses the very same Jacobi prior as its batch counterpart.

Construction is always double precision (``scipy``/``numpy``); ``dtype`` sets the stored precision.
"""
import abc
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.special import roots_hermite, roots_jacobi


def gauss_jacobi_nodes(alpha: float, beta: float, num_nodes: int,
                       dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss-Jacobi nodes (shifted to ``[0, 1]``) and normalized weights for a fixed ``Beta(alpha, beta)``.

    Computed once for a fixed prior (non-differentiable). The Jacobi weight on ``[-1, 1]`` is
    ``(1 - x)^(beta - 1) (1 + x)^(alpha - 1)``; the uniform ``Beta(1, 1)`` reduces to Gauss-Legendre.

    The rule is always *constructed* in double precision (``scipy.special.roots_jacobi``), which is
    where node/weight accuracy matters; ``dtype`` only sets the precision of the returned tensors.

    :return:
        theta_q   : nodes in ``[0, 1]``, shape ``(Q,)``
        log_v_q   : normalized log-weights, ``logsumexp_q = 0`` (``sum_q exp = 1``), shape ``(Q,)``
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    # mu=True returns the (analytical) weight sum, so we normalize in log space without a second pass.
    x, w, w_sum = roots_jacobi(num_nodes, beta - 1.0, alpha - 1.0, mu=True)  # weight (1-x)^(b-1)(1+x)^(a-1)
    theta_q = torch.as_tensor((x + 1.0) / 2.0, dtype=dtype)
    with np.errstate(divide="ignore"):  # a deep-tail weight underflowing to 0 -> -inf (node contributes 0)
        log_v_q = np.log(w) - np.log(w_sum)
    log_v_q = torch.as_tensor(log_v_q, dtype=dtype)
    return theta_q, log_v_q


def gauss_hermite_standard(num_nodes: int,
                           dtype: torch.dtype = torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardized Gauss-Hermite nodes and normalized log-weights for the weight ``e^{-x^2}``.

    Constructed in double precision via ``scipy.special.roots_hermite`` (preferred over numpy's
    ``hermgauss``: scipy switches to an asymptotic rule at large order, so it returns no NaN/overflow
    where ``hermgauss``'s ``1/fm^2`` weight step blows up — e.g. ``num_nodes >= ~400``). ``dtype`` sets
    the returned tensors' precision; the log representation keeps the tiny tail weights intact in float32
    (``log(1e-180) = -414`` stores fine where the weight itself underflows). Past where even float64
    ``w_i`` underflows (``num_nodes`` in the high hundreds) those weights map to ``-inf`` — a node that
    contributes nothing; recovering them would need a log-domain polynomial recurrence.

    :param
        num_nodes: number of quadrature nodes ``Q`` (>= 1)
        dtype: the output data type
    :return:
        x       : nodes, shape ``(Q,)``
        log_v   : normalized log-weights, ``logsumexp = 0`` (``sum exp = 1``), shape ``(Q,)``
    """
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1, got {num_nodes}")
    # mu=True returns the (analytical) weight sum sqrt(pi), so we normalize in log space inline.
    x, w, w_sum = roots_hermite(num_nodes, mu=True)  # physicists' Hermite: weight e^{-x^2}, sum(w)=sqrt(pi)
    x = torch.as_tensor(x, dtype=dtype)
    with np.errstate(divide="ignore"):  # deep-tail underflow (num_nodes in the high hundreds) -> -inf
        log_v = np.log(w) - np.log(w_sum)
    log_v = torch.as_tensor(log_v, dtype=dtype)
    return x, log_v


class FixedPrior(nn.Module, abc.ABC):
    """A fixed 1-D ability prior: two scalar parameters, broadcast to a batch.

    The base contract is just the prior parameters; priors that also carry a quadrature rule extend
    :class:`QuadraturePrior`. (The exact estimator needs only ``get_params``, so it takes a plain
    ``FixedPrior``.)
    """

    @abc.abstractmethod
    def get_params(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The two prior parameters, each broadcast to shape ``(batch_size,)``."""


class QuadraturePrior(FixedPrior):
    """A :class:`FixedPrior` bundled with its quadrature rule ``E_prior[f] = sum_q exp(log_v_q) f(theta_q)``.

    This is the contract a :class:`~betabern.core.model.estimator.QuadratureEstimator` depends on.
    Weights are carried in log space (``logsumexp_q log_v_q = 0``)
    """

    @abc.abstractmethod
    def quadrature(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch nodes ``theta_q`` (B, Q) and normalized log-weights ``log_v_q`` (B, Q)."""


class BetaPrior(FixedPrior):
    """Fixed ``Beta(alpha, beta)`` prior on ``theta in [0, 1]`` (parameters only, no quadrature)."""

    def __init__(self, alpha: float | int = 1.0, beta: float | int = 1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("beta", torch.tensor(float(beta)))

    def get_params(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.alpha.expand(batch_size).contiguous(), self.beta.expand(batch_size).contiguous()


class BetaJacobiPrior(BetaPrior, QuadraturePrior):
    """``Beta`` prior bundled with its Gauss-Jacobi quadrature (constant nodes; the prior is fixed)."""

    def __init__(self,
                 alpha: float | int = 1.0,
                 beta: float | int = 1.0,
                 num_nodes: int = 30,
                 dtype: torch.dtype = torch.float32):
        super().__init__(alpha, beta)

        theta_q, log_v_q = gauss_jacobi_nodes(alpha, beta, num_nodes, dtype=dtype)
        self.register_buffer("theta_q", theta_q)  # (Q,)
        self.register_buffer("log_v_q", log_v_q)  # (Q,) normalized log-weights

    def quadrature(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Constant nodes/log-weights broadcast to ``(B, Q)``."""
        return self.theta_q.expand(batch_size, -1), self.log_v_q.expand(batch_size, -1)


class NormalHermitePrior(QuadraturePrior):
    """Fixed ``Normal(mean, std)`` prior bundled with its (shifted) Gauss-Hermite quadrature."""

    def __init__(self,
                 mean: float | int = 0.0,
                 std: float | int = 1.0,
                 num_nodes: int = 30,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

        x, log_v = gauss_hermite_standard(num_nodes, dtype=dtype)
        self.register_buffer("gh_x", x)  # (Q,) standardized nodes
        self.register_buffer("gh_log_v", log_v)  # (Q,) normalized log-weights (shift-invariant)

    def get_params(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean.expand(batch_size), self.std.expand(batch_size)

    def quadrature(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized nodes shifted to ``N(mean, std^2)`` via ``theta = mean + std * sqrt(2) * x``."""
        mu, sigma = self.get_params(batch_size)
        theta_q = mu.reshape(-1, 1) + sigma.reshape(-1, 1) * math.sqrt(2.0) * self.gh_x.reshape(1, -1)  # (B, Q)
        log_v_q = self.gh_log_v.expand(batch_size, -1)  # (B, Q)
        return theta_q, log_v_q
