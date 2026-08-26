"""
Test suite for the online (static-trait, sequential) Beta-Bernstein quadrature estimator.

The online model mirrors the batch ``QuadratureEstimator``'s quadrature and processes responses one at
a time with NO learning transition, so it must be mathematically identical to the batch estimator over
the same IRF/prior. These tests pin that down:

  * the recurrent ``step`` loop (streaming form) equals the vectorized scan ``forward``;
  * the prequential per-step predictives sum to the batch marginal log-evidence (the chain rule);
  * the per-step posterior predictives equal the batch ``predictive_log_probs`` (on each growing prefix);
  * the final-step EAP equals the batch posterior mean (``posterior_stats``);
  * padding is ignored, predictives normalize over classes, and the posterior SD contracts with data;
  * the ``OnlineQuadTask`` training loss is the MML objective, and the pipeline runs end-to-end.
"""
import lightning as L
import torch

from betabern.bernstein.estimator.online_quad import (
    OnlineQuadEstimator,
    build_online_quad_estimator_free,
    build_online_quad_estimator_irt,
)
from betabern.bernstein.task import OnlineQuadTask
from betabern.core.data.datamodule import KnowledgeTracingDataModule, load_data
from betabern.core.model.estimator import QuadratureEstimator
from betabern.core.model.filter import BayesianFilter
from betabern.core.model.prior import BetaJacobiPrior

DATA_PATH = "data/simulated_student_log.csv"


# ===========================================================================
# Helpers
# ===========================================================================

def _make_batch(B=4, T=6, num_items=12, num_skills=3, seed=0, lengths=None) -> dict:
    """Full filter-style batch dict; ``lengths`` (per-row valid counts) enables ragged padding."""
    g = torch.Generator().manual_seed(seed)
    item_ids = torch.randint(0, num_items, (B, T), generator=g)
    skill_ids = torch.randint(0, num_skills, (B,), generator=g)
    answers = torch.randint(0, 2, (B, T), generator=g)
    user_ids = torch.randint(0, 5, (B,), generator=g)
    time_deltas = torch.zeros(B, T)

    if lengths is None:
        mask = torch.ones(B, T, dtype=torch.bool)
    else:
        mask = torch.arange(T)[None, :] < torch.as_tensor(lengths)[:, None]
        answers = answers.masked_fill(~mask, -1)
        item_ids = item_ids.masked_fill(~mask, -1)
    return dict(user_ids=user_ids, skill_ids=skill_ids, item_ids=item_ids,
                answers=answers, time_deltas=time_deltas, mask=mask)


def _online(num_nodes=30, degree=4, num_items=12, num_skills=3, kind="bkt", seed=0):
    torch.manual_seed(seed)
    build = build_online_quad_estimator_free if kind == "bkt" else build_online_quad_estimator_irt
    kwargs = dict(degree_n=degree, num_users=5, num_skills=num_skills, num_items=num_items,
                  num_nodes=num_nodes)
    if kind == "irt":
        kwargs.update(irt_model="2PL", item_weight_rank=1)
    return build(**kwargs)


def _batch_ref(model: OnlineQuadEstimator) -> QuadratureEstimator:
    """Batch quadrature estimator sharing the online model's IRF / prior / quadrature.

    Sharing the IRF instance (and recomputing the identical fixed nodes/basis) makes the two models
    one parameter set, so any difference in output is a difference in computation, not parameters.
    """
    prior = BetaJacobiPrior(model.prior.alpha.item(), model.prior.beta.item(), model.num_nodes)
    return QuadratureEstimator(prior=prior, irf=model.irf)


# ===========================================================================
# step-loop (streaming/recurrent form)  ==  vectorized forward (scan form)
# ===========================================================================

def test_step_loop_equals_vectorized_forward():
    """The recurrent ``step`` loop (``BayesianFilter.forward``) matches the vectorized scan ``forward``."""
    model = _online(kind="bkt")
    batch = _make_batch(B=4, T=6, lengths=[6, 5, 4, 5])

    vec_lp, vec_st = model.forward(**batch)  # vectorized scan
    seq_lp, seq_st = BayesianFilter.forward(model, **batch)  # sequential loop, drives step()

    assert vec_lp.shape == seq_lp.shape and vec_st.shape == seq_st.shape
    assert torch.allclose(vec_lp, seq_lp, atol=1e-4)
    assert torch.allclose(vec_st, seq_st, atol=1e-4)


# ===========================================================================
# Equivalence with the batch QuadratureEstimator (same parameters)
# ===========================================================================

def test_online_marginal_matches_batch():
    """Prequential sum_t log p(y_t | y_{<t}) equals the batch marginal log-evidence (chain rule)."""
    model = _online(kind="irt")
    est = _batch_ref(model)
    batch = _make_batch(B=5, T=5, lengths=[5, 5, 3, 4, 2])

    all_log_probs, _ = model.forward(**batch)  # (B, T, 2), masked steps zeroed
    ans = batch["answers"].clamp(min=0)
    obs_lp = all_log_probs.gather(2, ans[:, :, None]).squeeze(2) * batch["mask"]  # (B, T)
    online_evidence = obs_lp.sum(dim=1)  # (B,)

    batch_evidence = est.log_marginal_evidence(
        batch["skill_ids"], batch["item_ids"], batch["answers"], batch["mask"])
    assert torch.allclose(online_evidence, batch_evidence, atol=1e-3)


def test_online_predictive_matches_batch_prequential():
    """Online predictive at step t (given y_{<t}) equals the batch posterior-predictive on that prefix.

    The online model is prequential — each step conditions on every earlier response — so it matches
    the batch ``predictive_log_probs`` only when the latter is given the *same growing prefix* as
    history (not a fixed history for all target steps).
    """
    model = _online(kind="bkt")
    est = _batch_ref(model)
    seq = _make_batch(B=4, T=5, seed=7)  # fully valid
    online_lp, _ = model.forward(**seq)  # (B, T, 2)

    T = seq["item_ids"].size(1)
    for t in range(T):
        history = None if t == 0 else {
            "item_ids": seq["item_ids"][:, :t],
            "answers": seq["answers"][:, :t],
            "mask": seq["mask"][:, :t],
        }
        target = {"item_ids": seq["item_ids"][:, t:t + 1], "skill_ids": seq["skill_ids"],
                  "answers": seq["answers"][:, t:t + 1], "mask": seq["mask"][:, t:t + 1]}
        batch_step = est.predictive_log_probs(history, target)[:, 0, :]  # (B, 2) on prefix y_{<t}
        assert torch.allclose(online_lp[:, t, :], batch_step, atol=1e-4), f"mismatch at step {t}"


def test_online_final_eap_matches_batch():
    """The EAP read off the final state equals the batch posterior mean."""
    model = _online(kind="irt")
    est = _batch_ref(model)
    batch = _make_batch(B=5, T=5, lengths=[5, 4, 3, 5, 2])

    _, all_states = model.forward(**batch)
    online_eap, _ = model.eap_from_state(all_states[:, -1])  # (B,)

    batch_eap = est.posterior_stats(batch["skill_ids"], batch["item_ids"], batch["answers"], batch["mask"],
                                    level=None).mean
    assert online_eap.shape == (5,)
    assert torch.allclose(online_eap, batch_eap, atol=1e-4)


# ===========================================================================
# Distributional / structural properties
# ===========================================================================

def test_predictive_normalized_over_classes():
    """Per-step predictives are a valid distribution over the 2 classes at every valid step."""
    model = _online(kind="bkt")
    batch = _make_batch(B=4, T=5, lengths=[5, 4, 5, 3])
    all_log_probs, _ = model.forward(**batch)
    valid = all_log_probs[batch["mask"]]  # (Nvalid, 2)
    assert torch.allclose(torch.logsumexp(valid, dim=-1), torch.zeros(valid.size(0)), atol=1e-4)
    assert (valid <= 1e-4).all()


def test_padding_ignored():
    """Trailing padded steps change neither the marginal nor the final EAP."""
    model = _online(kind="bkt")
    est = _batch_ref(model)
    base = _make_batch(B=3, T=4, seed=3)  # fully valid

    pad = {k: v.clone() for k, v in base.items()}
    pad["item_ids"] = torch.cat([pad["item_ids"], torch.zeros(3, 2, dtype=torch.long)], dim=1)
    pad["answers"] = torch.cat([pad["answers"], torch.full((3, 2), -1, dtype=torch.long)], dim=1)
    pad["mask"] = torch.cat([pad["mask"], torch.zeros(3, 2, dtype=torch.bool)], dim=1)
    pad["time_deltas"] = torch.zeros(3, base["item_ids"].size(1) + 2)

    eap_base, _ = model.eap_from_state(model.forward(**base)[1][:, -1])
    eap_pad, _ = model.eap_from_state(model.forward(**pad)[1][:, -1])
    assert torch.allclose(eap_base, eap_pad, atol=1e-5)

    m_base = est.log_marginal_evidence(base["skill_ids"], base["item_ids"], base["answers"], base["mask"])
    m_pad = est.log_marginal_evidence(pad["skill_ids"], pad["item_ids"], pad["answers"], pad["mask"])
    assert torch.allclose(m_base, m_pad, atol=1e-5)


def test_posterior_sd_contracts_with_consistent_evidence():
    """With a stream of consistent (all-correct) responses the posterior concentrates: SD decreases."""
    model = _online(kind="irt")
    batch = _make_batch(B=4, T=8, seed=4)
    batch["answers"] = torch.ones_like(batch["answers"])  # all correct -> evidence piles up

    _, all_states = model.forward(**batch)
    _, sd = model.eap_from_state(all_states)  # (B, T + 1)
    assert (sd[:, -1] < sd[:, 0]).all()  # final tighter than the prior
    assert (sd[:, 1:] <= sd[:, :-1] + 1e-6).all()  # monotone non-increasing


# ===========================================================================
# Task layer
# ===========================================================================

def test_factories_build_online_estimator():
    """Both factories return a wired-up OnlineQuadEstimator with trainable shared params."""
    for kind in ("bkt", "irt"):
        model = _online(kind=kind)
        assert isinstance(model, OnlineQuadEstimator)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
        lp, st = model.forward(**_make_batch(B=2, T=3))
        assert lp.shape == (2, 3, 2) and st.shape == (2, 4, model.num_nodes)


def test_task_train_loss_is_mml_objective():
    """The inherited per-step NLL loss equals the (per-token) marginal-maximum-likelihood objective."""
    model = _online(kind="irt")
    task = OnlineQuadTask(model, learning_rate=0.01)
    batch = _make_batch(B=4, T=5, lengths=[5, 4, 3, 5])

    loss = task._compute_train_loss(batch)["loss"]
    n_valid = batch["mask"].sum()
    mml = _batch_ref(model).log_marginal_evidence(
        batch["skill_ids"], batch["item_ids"], batch["answers"], batch["mask"]).sum()

    assert loss.dim() == 0 and loss > 0
    assert torch.allclose(loss * n_valid, -mml, atol=1e-2)
    loss.backward()
    assert all(p.grad is not None for p in model.irf.parameters() if p.requires_grad)


def test_task_predict_step_tracks_eap():
    """predict_step returns an EAP trajectory in [0, 1] whose final value is the batch EAP."""
    model = _online(kind="bkt")
    task = OnlineQuadTask(model)
    batch = _make_batch(B=4, T=5, lengths=[5, 4, 5, 3])

    out = task.predict_step(batch, 0)
    B, T = batch["mask"].shape
    assert out["tracked_ability"].shape == (B, T + 1)
    assert (out["tracked_ability"] >= 0).all() and (out["tracked_ability"] <= 1).all()
    assert (out["tracked_variance"] >= 0).all()

    batch_eap = _batch_ref(model).posterior_stats(
        batch["skill_ids"], batch["item_ids"], batch["answers"], batch["mask"], level=None).mean
    assert torch.allclose(out["tracked_ability"][:, -1], batch_eap, atol=1e-4)


def test_trainer_fit_smoke():
    """The full DataModule + OnlineQuadTask + model pipeline survives a few Trainer steps."""
    df = load_data(DATA_PATH)
    dm = KnowledgeTracingDataModule(df, train_val_test_split=(0.7, 0.15, 0.15),
                                    split_strategy="within", batch_size=8)
    dm.setup(stage="fit")

    model = build_online_quad_estimator_irt(
        degree_n=4, num_users=dm.stats.num_users, num_skills=dm.stats.num_skills, num_items=dm.stats.num_items,
        num_nodes=24, irt_model="2PL", item_weight_rank=1)
    task = OnlineQuadTask(model, learning_rate=0.01)

    trainer = L.Trainer(max_epochs=2, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, logger=False,
                        limit_train_batches=3, limit_val_batches=2)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("\n✅ All online tests passed!")
