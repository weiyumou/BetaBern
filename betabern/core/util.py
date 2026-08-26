import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def raw_trainable_params(module: nn.Module) -> int:
    """Total ``requires_grad`` parameters — the raw storage count, before any free-DOF correction."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_free_params(module: nn.Module) -> int:
    """Number of *free* (identifiable) trainable parameters of a module.

    Defaults to :func:`raw_trainable_params`, but defers to a module's own ``num_free_params`` when it
    defines one — the hook for parameterizations whose stored tensors over-count the free degrees of
    freedom. The recurring case is a **softmax simplex**: it is shift-invariant, so ``k`` logits carry
    only ``k - 1`` identifiable DOF (and a model may pin further, e.g. 3PL fixes ``s = 0``). Composite
    modules (an estimator wrapping an IRF / IRT base) implement ``num_free_params`` by delegating to the
    sub-module that actually owns the parameters.

    Note: only *full* softmax embeddings carry a per-row shift redundancy; a low-rank ``A @ B`` factor
    generically cannot represent a per-row constant shift, so its weights are counted as storage.
    """
    counter = getattr(module, "num_free_params", None)
    return counter() if callable(counter) else raw_trainable_params(module)


def inv_softplus(x: float) -> float:
    """Numerically safe inverse softplus for non-negative scalar initializers."""
    # For large x, softplus(x) ≈ x and expm1(x) overflows.
    return x if x > 20.0 else math.log(math.expm1(x))


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0`` (Mächler 2012)."""
    return torch.where(x > -math.log(2.0),
                       torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def log_beta_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates the log of the Beta function B(x, y)"""
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)


def log_binom_coeff(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Calculates the log of the binomial coefficient comb(n, k)"""
    return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)


def gather_observed(log_per_class: torch.Tensor, answers: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Select each step's observed-class log-value and zero out padded steps.

    :param log_per_class: per-class log-values of shape ``(B, S, C, K)`` (C = response classes)
    :param answers: observed class indices ``(B, S)`` (negative padding is neutralized before gather)
    :param mask: validity mask ``(B, S)``; padded steps become 0 (a log-1 / no-op contribution)
    :return: the observed-class log-values, shape ``(B, S, K)``
    """
    K = log_per_class.size(-1)
    idx = answers.clamp(min=0)[:, :, None, None].expand(-1, -1, 1, K)  # (B, S, 1, K)
    obs = log_per_class.gather(2, idx).squeeze(2)  # (B, S, K)
    return obs * mask.unsqueeze(-1)


def log_matmul(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    """
    Performs matrix multiplication (M, K) @ (K, N) in log-space.
    :return: A tensor representing log(A @ B).
    """
    # Sum in log-space (equivalent to multiplication in linear space)
    # log_a: (M, K) -> (M, 1, K), log_b: (K, N) -> (1, N, K)
    # The result has shape (M, N, K) where element (i, j, k) is log(A_ik) + log(B_kj)
    sum_terms = log_a.unsqueeze(1) + log_b.T.unsqueeze(0)

    # Apply logsumexp along the K dimension
    # This is equivalent to summing the products in linear space
    log_c = torch.logsumexp(sum_terms, dim=-1)

    return log_c  # (M, N)


class LowRankEmbedding(nn.Module):
    """Embedding layer with optional low-rank factorization.

    Parameters
    ----------
    num_embedding : int
        Number of embeddings (vocabulary size).
    embedding_dim : int
        Dimension of each embedding vector.
    rank : int | None
        If None, always return zeros (no learnable parameters).
        If <= 0, use full-rank learnable embeddings.
        Otherwise, factorize as A @ B with A: (num_embedding, rank), B: (rank, embedding_dim).
    """

    def __init__(self, num_embedding: int, embedding_dim: int, rank: int | None = -1):
        super().__init__()

        if rank is None:  # No learnable parameters; always return zeros
            self.register_buffer("_weight", torch.zeros(num_embedding, embedding_dim))
            self.A = self.B = None
        else:
            rank = min(num_embedding, embedding_dim, rank)
            if (rank <= 0) or (rank == embedding_dim) or (rank == num_embedding):  # Full rank; no approximation
                self._weight = nn.Parameter(torch.zeros(num_embedding, embedding_dim))
                self.A = self.B = None
            else:
                self._weight = None
                std_dev = 1 / math.sqrt(rank)
                self.A = nn.Parameter(torch.randn(num_embedding, rank) * std_dev)
                self.B = nn.Parameter(torch.zeros(rank, embedding_dim))

    @property
    def weight(self) -> torch.Tensor:
        if self._weight is not None:
            return self._weight
        return self.get_parameter("A") @ self.get_parameter("B")

    def forward(self, indices: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return F.embedding(indices, self.weight, *args, **kwargs)
