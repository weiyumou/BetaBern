import abc

import torch
import torch.nn as nn

EPS = 1e-8


class BayesianFilter(nn.Module, abc.ABC):
    """Sequential latent-state filter over a response sequence.

    The default :meth:`forward` runs a generic left-to-right loop that calls :meth:`step` at each
    timestep (carrying a fixed-shape, masked state) — the right execution model for filters whose
    state transition is nonlinear / non-associative and therefore cannot be vectorized over time.

    Extension points:
    - ``get_init_state`` (**required**): the initial per-student state.
    - ``step`` (**optional**): the per-step recurrence the default ``forward`` loop calls; a filter
      that relies on that loop must implement it. A filter whose recurrence is an *associative scan*
      (e.g. a transition-free additive update) may instead override ``forward`` with a vectorized pass
      and omit ``step`` — implementing it only to expose a one-response-at-a-time streaming interface.
      See ``bernstein/online/quad.py``.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_init_state(self, skill_ids: torch.Tensor) -> torch.Tensor:
        """Return the initial state of the Bayesian filter for the given skill IDs."""

    def step(self,
             state: torch.Tensor,
             user_t: torch.Tensor,
             skill_t: torch.Tensor,
             item_t: torch.Tensor,
             answer_t: torch.Tensor,
             delta_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform a single step of the Bayesian filter, returning the batch log-probability and next state.

        Optional hook (not abstract): required only by the default sequential :meth:`forward` loop.
        Filters that override ``forward`` with a vectorized/scan pass need not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement step(); implement it to use the default "
            f"sequential forward() loop, or override forward() directly.")

    def forward(
            self,
            user_ids: torch.Tensor,
            skill_ids: torch.Tensor,
            item_ids: torch.Tensor,
            answers: torch.Tensor,
            time_deltas: torch.Tensor,
            mask: torch.Tensor,
            current_state: torch.Tensor | None = None,
    ):
        """Run the full sequential forward pass.

        Returns
        -------
        all_log_probs : (B, S, C) log p(y_t | history) per timestep for each response class
        all_states : (B, S + 1, 2) state trajectory
        """
        # Initialize state
        if current_state is None:
            current_state = self.get_init_state(skill_ids)  # (B, 2)

        batch_size, seq_len = answers.size()
        # Typically data is organized by (user, skill), but this allows grouping by user as well
        if skill_ids.dim() == 1:
            skill_ids = skill_ids.unsqueeze(-1).expand(-1, seq_len)

        all_log_probs = []
        all_states = [current_state]
        for t in range(seq_len):
            # Determine active batch elements at this timestep
            mask_t = mask[:, t]
            log_probs, next_state = self.step(state=current_state[mask_t],
                                              user_t=user_ids[mask_t],
                                              skill_t=skill_ids[mask_t, t],
                                              item_t=item_ids[mask_t, t],
                                              answer_t=answers[mask_t, t],
                                              delta_t=time_deltas[mask_t, t])

            # Write back
            curr_log_p = torch.zeros((batch_size, log_probs.size(-1)),
                                     device=log_probs.device, dtype=log_probs.dtype)
            curr_log_p[mask_t] = log_probs
            all_log_probs.append(curr_log_p)

            current_state = current_state.clone()
            current_state[mask_t] = next_state
            all_states.append(current_state)

        all_log_probs = torch.stack(all_log_probs, dim=1)  # (B, T, C)
        all_states = torch.stack(all_states, dim=1)  # (B, T + 1, 2)

        return all_log_probs, all_states


class LearningRateGenerator(nn.Module):
    def __init__(self, num_users: int, num_skills: int):
        super().__init__()

        # self.user_lr_logit = nn.Embedding(num_users, 1)
        # nn.init.constant_(self.user_lr_logit.weight, torch.logit(torch.tensor(0.01)).item())

        self.skill_lr_logit = nn.Embedding(num_skills, 1)
        nn.init.constant_(self.skill_lr_logit.weight, torch.logit(torch.tensor(0.01)).item())

    def forward(self, user_ids: torch.Tensor, skill_ids: torch.Tensor) -> torch.Tensor:
        # user_lr = self.user_logit_lr(user_ids)  # (B, 1)
        skill_lr = self.skill_lr_logit(skill_ids)  # (B, 1)
        return torch.sigmoid(skill_lr)  # (B, 1)
