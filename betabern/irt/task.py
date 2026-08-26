import torch

from betabern.core.data.datamodule import is_stacked_batch
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.prior import NormalHermitePrior
from betabern.core.task import KnowledgeTracingTask
from betabern.irt.estimator.static_irt import StaticIRT
from betabern.irt.model import LogisticIRF

# ======================================================================
# StaticIRT Task
# ======================================================================

class StaticIRTTask(KnowledgeTracingTask):
    """Lightning task for StaticIRT (joint MLE of items + per-student abilities).

    Training calibrates the items and learns each training student's θ embedding (with a MAP prior).
    At eval, a **stacked** ``(history, target)`` batch scores held-out students (absent from the
    embedding) against the frozen items by **EAP**: the trained items (as a :class:`LogisticIRF`) are run
    through a :class:`~betabern.core.model.estimator.QuadratureEstimator` view (Normal prior +
    Gauss-Hermite, no optimization), so scoring is identical to the IRT MML estimator and a
    StaticIRT-vs-IRT-MML comparison isolates *item calibration* (JML vs MML). A non-stacked batch uses
    the learned embedding (calibration students).

    Parameters
    ----------
    model : StaticIRT
    learning_rate : float
    theta_reg : float
        L2 (Gaussian-prior) strength on the θ embedding during training.
    eap_num_nodes : int
        Gauss-Hermite node count for EAP scoring (default ``N(0, 1)``, matching the IRT MML estimator's
        default for an apples-to-apples comparison).
    """

    def __init__(self,
                 model: StaticIRT,
                 learning_rate: float = 0.02,
                 theta_reg: float = 1.0,
                 eap_num_nodes: int = 30):
        super().__init__(model=model, learning_rate=learning_rate)

        self.theta_reg = theta_reg
        # posterior scorer reuses the model's items. Quadrature over a fixed Normal prior -> no optimization needed.
        self._scorer = QuadratureEstimator(prior=NormalHermitePrior(0.0, 1.0, eap_num_nodes), irf=LogisticIRF(model))

    def _compute_train_loss(self, batch) -> dict:
        results = super()._compute_train_loss(batch)

        # MAP prior on θ: -log N(θ|0, 1/λ) ∝ (λ/2)||θ||²
        target_batch = results["target_batch"]
        theta = self.model.get_theta(target_batch["user_ids"], target_batch["skill_ids"])  # (B, 1)
        results["loss"] = results["loss"] + 0.5 * self.theta_reg * (theta ** 2).mean()

        return results

    def forward(self, batch: dict) -> dict:
        if not is_stacked_batch(batch):
            # Calibration students — use the learned embedding θ.
            all_log_probs = self.model(user_ids=batch["user_ids"],
                                       skill_ids=batch["skill_ids"],
                                       item_ids=batch["item_ids"])  # (B, S, 2)
            return dict(all_log_probs=all_log_probs, target_batch=batch, history_batch=None)

        # Held-out students: EAP posterior-predictive from the history against the frozen items.
        history, target_batch = batch
        all_log_probs = self._scorer.predictive_log_probs(history, target_batch)  # (B, S, 2)
        return dict(all_log_probs=all_log_probs, target_batch=target_batch, history_batch=history)

    def predict_step(self, batch, batch_idx):
        results = self.forward(batch)
        all_log_probs, target_batch = results["all_log_probs"], results["target_batch"]
        history = results["history_batch"]

        if history is not None:  # held-out students: posterior mean + variance over the history
            post = self._scorer.posterior_stats(history["skill_ids"], history["item_ids"],
                                                history["answers"], history["mask"], level=None)  # no CI
            theta, var = post.mean.unsqueeze(-1), post.variance.unsqueeze(-1)  # (B, 1)
        else:  # calibration students -> learned embedding (a JML point estimate, no posterior variance)
            theta = self.model.get_theta(target_batch["user_ids"], target_batch["skill_ids"])  # (B, 1)
            var = torch.zeros_like(theta)
        theta = theta.expand_as(target_batch["item_ids"])  # (B, S)
        var = var.expand_as(target_batch["item_ids"])  # (B, S)

        return {
            "log_p_correct": all_log_probs[:, :, 1],
            "answers": target_batch["answers"],
            "item_ids": target_batch["item_ids"],
            "tracked_ability": theta,
            "tracked_variance": var,
            "mask": target_batch["mask"],
            "user_ids": target_batch["user_ids"],
            "skill_ids": target_batch["skill_ids"],
        }
