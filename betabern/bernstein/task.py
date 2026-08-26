import torch

from betabern.core.task import BayesianFilterTask

# ======================================================================
# OnlineQuadEstimator Task (static-trait, prequential)
# ======================================================================

class OnlineQuadTask(BayesianFilterTask):
    """Filter task for the online (static-trait) ``OnlineQuadEstimator``.

    With no learning transition, the inherited per-step NLL training loss equals the prequential
    log-loss and hence the marginal-maximum-likelihood objective (chain rule). The filter state is the
    per-node cumulative log-likelihood ``S_q``; the tracked ability is the posterior EAP read off it.
    """

    def _tracked_ability(self, all_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eap, sd = self.model.eap_from_state(all_states)  # each (B, S + 1)
        return eap, sd ** 2
