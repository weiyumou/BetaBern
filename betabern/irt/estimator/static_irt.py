"""
Parametric IRT Models (1PL - 4PL) in Plain PyTorch.

These models support:
- Per-user, per-skill abilities (θ[user, skill])
- Joint MLE training of item and ability parameters
- MAP regularization on abilities
"""

import torch
import torch.nn as nn

from betabern.irt.model import IRTBase


class StaticIRT(IRTBase):
    """
    Static IRT model supporting 1PL through 4PL.
    It assumes a static ability throughout interactions.

    Parameters
    ----------
    num_users : int
        Number of users.
    num_skills : int
        Number of skills.
    num_items : int
        Number of items.
    irt_model : str
        IRT model type: "1PL", "2PL", "3PL", or "4PL".
    c_init : float
        Initial guessing probability (3PL/4PL only).
    s_init : float
        Initial slipping probability (4PL only).
    """

    def __init__(
            self,
            num_users: int,
            num_skills: int,
            num_items: int,
            irt_model: str = "2PL",
            c_init: float = 0.1,
            s_init: float = 0.1,
            item_weight_rank: int | None = None,
    ):
        super().__init__(
            num_users=num_users,
            num_skills=num_skills,
            num_items=num_items,
            irt_model=irt_model,
            c_init=c_init,
            s_init=s_init,
            enable_item_params=(item_weight_rank is not None),  # any rank -> item params on
        )

        # --- User-Skill Abilities ---
        # Shape: (num_users, num_skills)
        self.theta = nn.Embedding(num_users, num_skills)
        nn.init.zeros_(self.theta.weight)

    def get_theta(self, user_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up abilities for user-skill pairs.

        Parameters
        ----------
        user_ids : Tensor
            Shape (batch_size,)
        skill_ids : Tensor
            Shape (batch_size,)

        Returns
        -------
        Tensor
            Shape (batch_size, 1)
        """
        # Get all abilities for these users: (batch_size, num_skills)
        user_abilities = self.theta(user_ids)

        # Select the skill-specific ability: (batch_size, 1)
        theta = user_abilities.gather(1, skill_ids.unsqueeze(-1))

        return theta

    def forward(
            self,
            user_ids: torch.Tensor,
            skill_ids: torch.Tensor,
            item_ids: torch.Tensor,
            theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute log P(correct) and log P(incorrect) for each response.

        Parameters
        ----------
        user_ids : Tensor
            Shape (batch_size,) - one user per sequence
        skill_ids : Tensor
            Shape (batch_size,) - one skill per sequence
        item_ids : Tensor
            Shape (batch_size, seq_len) - padded item IDs
        theta : Tensor, optional
            Precomputed per-student abilities of shape (batch_size, 1). If None, looks them up from
            user_ids and skill_ids.

        Returns
        -------
        A Tensor of shape (batch_size, seq_len, 2)
        """
        if theta is None:
            # Get user-skill ability: (batch_size, 1)
            theta = self.get_theta(user_ids, skill_ids)

        # Static ability is a single shared node (Q = 1); log_irf evaluates the IRF at it for every item
        # (and expands the per-sequence skill_ids internally).
        return self.log_irt(theta, item_ids, skill_ids).squeeze(-1)  # (B, S, 2)
