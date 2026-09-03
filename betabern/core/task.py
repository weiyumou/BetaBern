"""
Knowledge Tracing Tasks (Lightning Modules)

Model-agnostic PyTorch Lightning wrappers: a shared metrics/training base
(:class:`KnowledgeTracingTask`) plus the two families — :class:`BayesianEstimatorTask` (static,
marginal-likelihood) and :class:`BayesianFilterTask` (sequential). Model-specific read-outs live in
the per-package subclasses.
"""
import abc

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.regression import MeanSquaredError

from betabern.core.data.datamodule import is_stacked_batch

TRAIN_PREFIX = "1-train"
VAL_PREFIX = "2-val"
TEST_PREFIX = "3-test"


class KnowledgeTracingTask(L.LightningModule, abc.ABC):
    """
    Base Lightning Module for training and evaluating knowledge tracing models.

    Parameters
    ----------
    model : nn.Module
        The model to train: a ``BayesianEstimator``, a ``BayesianFilter``, or a ``StaticIRT``.
    learning_rate : float
        Learning rate for the optimizer.
    """

    def __init__(self, model, learning_rate: float = 0.02, monitor: str = f"1-log-loss/{VAL_PREFIX}"):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        # Metric the LR scheduler reduces on. Defaults to the validation log-loss; callers that train
        # without a validation split (see :func:`betabern.core.harness.fit`) point it at the
        # training loss instead, since the val metric is never logged.
        self.monitor = monitor
        self.save_hyperparameters(ignore=["model"])

        # Stateful metrics that accumulate across batches for correct global computation
        metrics = MetricCollection({
            "acc": BinaryAccuracy(),
            "auc": BinaryAUROC(),
            "rmse": MeanSquaredError(squared=False),
        })
        self.val_metrics = metrics.clone(prefix=f"{VAL_PREFIX}/")
        self.test_metrics = metrics.clone(prefix=f"{TEST_PREFIX}/")

    @abc.abstractmethod
    def forward(self, batch) -> dict:
        """
        At a minimum, it should return a dict with two keys:
        - all_log_probs: all log probabilities evaluated for the target batch (shape [B, S, 2])
        - target_batch: the target batch of data
        Other keys may be included if appropriate for specific models.
        """

    def _compute_train_loss(self, batch) -> dict:
        """
        Returns a dictionary containing at least the key "loss" for the training loss computed on the batch.
        """
        results = self.forward(batch)
        all_log_probs, target_batch = results["all_log_probs"], results["target_batch"]
        answers, mask = target_batch["answers"], target_batch["mask"]
        results["loss"] = F.nll_loss(all_log_probs[mask], answers[mask], reduction="mean")
        return results

    def training_step(self, batch, batch_idx):
        loss = self._compute_train_loss(batch)["loss"]
        self.log_dict({f"1-log-loss/{TRAIN_PREFIX}": loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            ),
            "monitor": self.monitor,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_eval_step(self, batch, prefix: str):
        results = self.forward(batch)
        all_log_probs, target_batch = results["all_log_probs"], results["target_batch"]
        answers, mask = target_batch["answers"], target_batch["mask"]
        flat_labels = answers[mask]
        loss = F.nll_loss(all_log_probs[mask], flat_labels, reduction="mean")

        # log P(correct) is index 1 in the last dimension
        flat_probs = torch.exp(all_log_probs[mask][:, 1])

        # Log loss per-batch (mean reduction is fine for NLL)
        self.log(f"1-log-loss/{prefix}", loss, prog_bar=(prefix == VAL_PREFIX))

        # Update stateful metrics (accumulated across batches, computed at epoch end)
        metrics = self.val_metrics if prefix == VAL_PREFIX else self.test_metrics
        metrics["acc"].update(flat_probs, flat_labels)
        metrics["auc"].update(flat_probs, flat_labels)
        metrics["rmse"].update(flat_probs, flat_labels.float())

    def _log_epoch_metrics(self, metrics: MetricCollection, prefix: str):
        """Compute and log accumulated metrics at epoch end."""
        results = metrics.compute()
        self.log_dict({
            f"2-acc/{prefix}": results[f"{prefix}/acc"],
            f"3-auc/{prefix}": results[f"{prefix}/auc"],
            f"4-rmse/{prefix}": results[f"{prefix}/rmse"],
        })
        metrics.reset()

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, prefix=VAL_PREFIX)

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(self.val_metrics, prefix=VAL_PREFIX)

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, prefix=TEST_PREFIX)

    def on_test_epoch_end(self):
        self._log_epoch_metrics(self.test_metrics, prefix=TEST_PREFIX)


# ======================================================================
# Estimator Task (Marginal Maximum Likelihood) — model-agnostic
# ======================================================================

class BayesianEstimatorTask(KnowledgeTracingTask):
    """Lightning task for any :class:`BayesianEstimator`.

    Training maximizes the marginal log-evidence per student (the latent ability is integrated out
    by the model). Evaluation uses the posterior-predictive distribution on the target responses
    given the observed history (or the prior, when no history is provided). Depends only on the
    ``BayesianEstimator`` contract, so it works for any estimator regardless of how the marginal
    integral is solved.
    """

    def _compute_train_loss(self, batch) -> dict:
        # MML objective: the full sequence is the response vector whose ability is marginalized.
        target_batch = batch[-1] if is_stacked_batch(batch) else batch
        mask = target_batch["mask"]
        log_evidence = self.model.log_marginal_evidence(
            skill_ids=target_batch["skill_ids"],
            item_ids=target_batch["item_ids"],
            answers=target_batch["answers"],
            mask=mask,
        )
        # Per-token scaling: a constant rescale of the true MML objective (optimum unchanged) that
        # keeps the loss magnitude stable and comparable across batches.
        loss = -log_evidence.sum() / mask.sum().clamp(min=1)
        irf = getattr(self.model, "irf", None)  # optional IRF-level regularizer (e.g. soft-monotonicity prior)
        if irf is not None and hasattr(irf, "weight_penalty"):
            loss = loss + irf.weight_penalty()
        return {"loss": loss, "target_batch": target_batch}

    def forward(self, batch) -> dict:
        if is_stacked_batch(batch):
            history_batch, target_batch = batch
        else:
            history_batch, target_batch = None, batch

        all_log_probs = self.model.predictive_log_probs(history_batch, target_batch)  # (B, S, 2)
        return dict(all_log_probs=all_log_probs, target_batch=target_batch, history_batch=history_batch)

    def predict_step(self, batch, batch_idx):
        results = self.forward(batch)
        all_log_probs, target_batch = results["all_log_probs"], results["target_batch"]

        # Estimate static ability (posterior mean + variance) from whatever responses are observed;
        # level=None skips the (expensive) credible interval, which prediction does not need.
        observed = results["history_batch"] if results["history_batch"] is not None else target_batch
        post = self.model.posterior_stats(
            skill_ids=observed["skill_ids"],
            item_ids=observed["item_ids"],
            answers=observed["answers"],
            mask=observed["mask"],
            level=None,
        )

        tracked_ability = post.mean.reshape(-1, 1).expand_as(target_batch["item_ids"])  # (B, S)
        tracked_variance = post.variance.reshape(-1, 1).expand_as(target_batch["item_ids"])  # (B, S)
        return {
            "log_p_correct": all_log_probs[..., 1],
            "answers": target_batch["answers"],
            "item_ids": target_batch["item_ids"],
            "tracked_ability": tracked_ability,
            "tracked_variance": tracked_variance,
            "mask": target_batch["mask"],
            "user_ids": target_batch["user_ids"],
            "skill_ids": target_batch["skill_ids"],
        }


# ======================================================================
# Filter Task (sequential) — model-agnostic
# ======================================================================

class BayesianFilterTask(KnowledgeTracingTask):
    """Lightning task for any :class:`~betabern.core.model.filter.BayesianFilter`.

    Runs the filter over the sequence (continuing from the history's final state for stacked batches)
    and trains on the per-step NLL inherited from :class:`KnowledgeTracingTask`. The only
    model-specific part is how the state trajectory maps to a tracked ability, supplied by subclasses
    via :meth:`_tracked_ability`.
    """

    def forward(self, batch: dict) -> dict:
        # Continue from the history's final state for stacked (history, target) batches.
        current_state = None
        if is_stacked_batch(batch):
            history_batch, target_batch = batch
            with torch.inference_mode():
                _, all_states = self.model(**history_batch)
            current_state = all_states[:, -1]
        else:
            target_batch = batch

        all_log_probs, all_states = self.model(**target_batch, current_state=current_state)
        return dict(all_log_probs=all_log_probs, all_states=all_states, target_batch=target_batch)

    @abc.abstractmethod
    def _tracked_ability(self, all_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map the state trajectory to ``(tracked_ability, tracked_variance)``, each ``(B, S + 1)``."""

    def predict_step(self, batch, batch_idx):
        results = self.forward(batch)
        all_log_probs, all_states = results["all_log_probs"], results["all_states"]
        target_batch = results["target_batch"]

        ability, variance = self._tracked_ability(all_states)
        return {
            "log_p_correct": all_log_probs[..., 1],
            "answers": target_batch["answers"],
            "item_ids": target_batch["item_ids"],
            "tracked_ability": ability,
            "tracked_variance": variance,
            "all_states": all_states,
            "mask": target_batch["mask"],
            "user_ids": target_batch["user_ids"],
            "skill_ids": target_batch["skill_ids"],
        }
