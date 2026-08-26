import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from betabern.core.model.irf import MonotoneIRF
from betabern.core.util import inv_softplus, raw_trainable_params

EPS = 1e-8


@dataclass
class IRTParams:
    a: torch.Tensor  # Discrimination parameter
    b: torch.Tensor  # Difficulty parameter
    log_c: torch.Tensor  # Guessing parameter in log space
    log_s: torch.Tensor  # Slipping parameter in log space
    log_m: torch.Tensor  # Margin parameter in log space (m = 1 - c - s)

    def __post_init__(self):
        # Calculate attribute after the object is created
        self.c = torch.exp(self.log_c)
        self.s = torch.exp(self.log_s)


class IRTBase(nn.Module):

    def __init__(self,
                 num_users: int,
                 num_skills: int,
                 num_items: int,
                 irt_model: str = "1PL",
                 c_init: float = 0.1,
                 s_init: float = 0.1,
                 enable_item_params: bool = False):
        super().__init__()

        # The model code carries both the parameter count and the link (logistic vs. normal ogive),
        # e.g. "2PL" -> (2, logistic), "2PO" -> (2, ogive).
        code = irt_model.upper()
        if (m := re.fullmatch(r"([1-4])(PL|PO)", code)) is None:
            raise ValueError(f"irt_model must be like '2PL' (logistic) or '2PO' (normal ogive), not '{irt_model}'")
        self.irt_model = code
        self.n_params, self.link = int(m.group(1)), ("logistic" if m.group(2) == "PL" else "ogive")

        self.num_users = num_users
        self.num_skills = num_skills
        self.num_items = num_items
        self.enable_item_params = enable_item_params

        # --- Difficulty Parameters (b) ---
        self.item_b = nn.Embedding(num_items, 1)
        self.skill_b = nn.Embedding(num_skills, 1)

        # --- Discrimination Parameters (a) ---
        self.item_a_raw = nn.Embedding(num_items, 1)
        self.skill_a_raw = nn.Embedding(num_skills, 1)

        # --- Endpoint Logits for [c, s, m] ---
        self.item_endpoint_logits = nn.Embedding(num_items, 3)
        self.skill_endpoint_logits = nn.Embedding(num_skills, 3)

        self._initialize_params(c_init, s_init)

    @torch.no_grad()
    def _initialize_params(self, c_init: float, s_init: float):
        if c_init + s_init >= 1.0:
            raise ValueError("The sum of c_init and s_init cannot be larger than 1")

        learn_a = self.n_params >= 2  # discrimination
        learn_c = self.n_params >= 3  # guessing
        learn_s = self.n_params >= 4  # slipping

        # --- Difficulty Parameters (b) ---
        self.item_b.requires_grad_(self.enable_item_params)
        nn.init.zeros_(self.item_b.weight)
        nn.init.zeros_(self.skill_b.weight)

        # --- Discrimination Parameters (a) ---
        self.item_a_raw.requires_grad_(self.enable_item_params and learn_a)
        nn.init.zeros_(self.item_a_raw.weight)

        self.skill_a_raw.requires_grad_(learn_a)
        nn.init.constant_(self.skill_a_raw.weight, inv_softplus(1.0))

        # --- Endpoint Logits for [c, s, m] ---
        c_val = c_init if learn_c else 0.0
        s_val = s_init if learn_s else 0.0
        m_val = 1.0 - c_val - s_val

        # Skill endpoints carry the init; item endpoints are a zero-initialized per-item offset
        init_log_probs = torch.log(torch.tensor([c_val, s_val, m_val]) + EPS)
        self.skill_endpoint_logits.weight.copy_(
            init_log_probs.unsqueeze(0).expand(self.num_skills, -1)
        )
        nn.init.zeros_(self.item_endpoint_logits.weight)

        self.skill_endpoint_logits.requires_grad_(learn_c or learn_s)
        self.item_endpoint_logits.requires_grad_(self.enable_item_params and (learn_c or learn_s))

    def num_free_params(self) -> int:
        """Number of *free* IRT parameters (the ``num_free_params`` protocol; see ``core.util``).

        The [c, s, m] endpoints are a 3-logit softmax, so not every stored logit is a free DOF — the
        raw ``parameters()`` count over-counts them. Subtract the redundant logits: per skill (the skill
        endpoints are always learned at 3+ params) and, when item params are on, per item. Link-agnostic
        (logistic and normal-ogive share the parameter structure).
        """
        total = raw_trainable_params(self)
        match self.n_params:
            case 3:  # only c is free (s pinned to 0, m = 1 - c): 3 logits -> 1 DOF, 2 redundant
                total -= self.num_skills * 2
                if self.enable_item_params:
                    total -= self.num_items * 2
            case 4:  # c, s free (m = 1 - c - s): 3 logits -> 2 DOF, 1 redundant
                total -= self.num_skills
                if self.enable_item_params:
                    total -= self.num_items
        return total

    def get_irt_params(self, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> IRTParams:
        """Get IRT parameters for specific items and skills."""
        # Clamp item_ids and skill_ids to valid range (handle padding values like -1)
        item_ids = torch.clamp(item_ids, 0, self.num_items - 1)
        skill_ids = torch.clamp(skill_ids, 0, self.num_skills - 1)

        a = F.softplus(self.item_a_raw(item_ids) + self.skill_a_raw(skill_ids))
        b = self.item_b(item_ids) + self.skill_b(skill_ids)

        # [c, s, m] via softmax
        logits = self.item_endpoint_logits(item_ids) + self.skill_endpoint_logits(skill_ids)
        log_endpoints = F.log_softmax(logits, dim=-1)
        log_c, log_s, log_m = torch.split(log_endpoints, 1, dim=-1)

        return IRTParams(a=a, b=b, log_c=log_c, log_s=log_s, log_m=log_m)

    def log_irt(self, theta: torch.Tensor, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Per-node IRT response log-likelihood: the (1-4PL) IRF evaluated at a set of ability nodes.

        ``P = c + m * F(a(theta - b))``, ``1 - P = s + m * F(-a(theta - b))``, in log-space, where the link
        ``F`` is the logistic sigmoid (``2PL``) or the standard-normal CDF (``2PO``), per ``self.link``.
        The single response read-out for the package — a per-student ability is just the ``Q = 1`` case
        (see ``StaticIRT.forward``). Mirrors ``LogisticBernsteinIRF.log_irf`` but with the raw IRT curve, so
        it can also serve as the per-node likelihood of a quadrature-based estimator.

        :param theta: ability nodes of shape ``(B, Q)`` (one set of Q nodes per student)
        :param item_ids: item indices of shape ``(B, S)``
        :param skill_ids: skill indices of shape ``(B,)`` or ``(B, S)``
        :return: log probabilities of shape ``(B, S, 2, Q)`` (ordered [incorrect, correct])
        """
        if skill_ids.dim() == 1:
            skill_ids = skill_ids.unsqueeze(-1).expand_as(item_ids)  # (B, S)

        params = self.get_irt_params(item_ids, skill_ids)  # each (B, S, 1)
        logit = params.a * (theta.unsqueeze(dim=1) - params.b)  # (B, S, Q): Q nodes broadcast across items
        # log link CDF: logistic -> log-sigmoid; ogive (normal) -> log standard-normal CDF (log_ndtr).
        log_cdf = F.logsigmoid if self.link == "logistic" else torch.special.log_ndtr
        # log(c + m * F(logit))  and  log(s + m * F(-logit) = s + m * (1 - F(logit))), via logaddexp in log-space
        log_p_correct = torch.logaddexp(params.log_c, params.log_m + log_cdf(logit))
        log_p_incorrect = torch.logaddexp(params.log_s, params.log_m + log_cdf(-logit))
        return torch.stack([log_p_incorrect, log_p_correct], dim=2)  # (B, S, 2, Q)


class LogisticIRF(MonotoneIRF):
    """The (1-4PL) IRT IRF on ``theta in R``, backed by an :class:`IRTBase`.

    The link (logistic sigmoid or normal ogive) is set by the wrapped :class:`IRTBase`'s ``irt_model`` code
    (e.g. ``2PL`` vs. ``2PO``); this wrapper is link-agnostic and just delegates to :meth:`IRTBase.log_irt`.
    """

    def __init__(self, irt_base: IRTBase):
        super().__init__()
        self.irt_base = irt_base

    def num_free_params(self) -> int:
        """All learnable parameters live in the wrapped IRT base."""
        return self.irt_base.num_free_params()

    def log_irf(self, theta: torch.Tensor, item_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """Log logistic IRF at ability nodes ``theta`` (B, Q) -> ``(B, S, 2, Q)`` ([incorrect, correct])."""
        return self.irt_base.log_irt(theta, item_ids, skill_ids)


def make_logistic_irf(num_users: int,
                      num_skills: int,
                      num_items: int,
                      irt_model: str = "2PL",
                      c_init: float = 0.1,
                      s_init: float = 0.1,
                      item_weight_rank: int | None = None) -> LogisticIRF:
    """Fresh IRT IRF backed by a new :class:`IRTBase` (``irt_model`` selects logistic ``2PL`` vs. ogive ``2PO``)."""
    irt_base = IRTBase(num_users=num_users, num_skills=num_skills, num_items=num_items,
                       irt_model=irt_model, c_init=c_init, s_init=s_init,
                       enable_item_params=(item_weight_rank is not None))  # any rank -> item params on
    return LogisticIRF(irt_base)
