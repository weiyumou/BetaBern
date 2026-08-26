"""``betabern fit`` — fit an estimator to response data, write an annotated prediction CSV.

Examples
--------
# Exact Beta-Bernstein estimator (IRT-parameterized weights) on a response-log CSV:
    betabern fit --data data/simulated_student_log.csv --model bern-exact --degree 10

# Free-form weights, Gauss-Jacobi quadrature marginalization:
    betabern fit --data ... --model bern-quad --weights free

# Sequential (online, static-trait) estimator; same MML objective, left-to-right:
    betabern fit --data ... --model bern-online --weights free

# Gauss-Hermite IRT MML baseline / JML StaticIRT baseline:
    betabern fit --data ... --model irt-mml --irt-model 3PL
    betabern fit --data ... --model irt-jml

# Straight from an Item Response Warehouse table (cached under data/irw/, fetched when missing):
    betabern fit --irw art --model bern-exact
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import lightning as L

from betabern.core import harness
from betabern.core.data import ingest
from betabern.core.task import BayesianEstimatorTask

MODELS = ("bern-exact", "bern-quad", "bern-online", "irt-mml", "irt-jml")


def add_arguments(p: argparse.ArgumentParser) -> None:

    # Data (one of --data / --irw)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Response-log CSV (user_id, skill_name, item_id, is_correct"
                                    "[, timestamp]) or an IRW-shaped CSV (id, item, resp).")
    src.add_argument("--irw", help="Item Response Warehouse table name (cached under data/irw/; "
                                   "fetched on first use — needs the 'irw' package + Redivis token).")
    p.add_argument("--skills", nargs="*", default=None, help="Optional subset of skill names.")

    # Model
    p.add_argument("--model", required=True, choices=MODELS,
                   help="bern-exact (closed-form marginal likelihood), bern-quad (Gauss-Jacobi), "
                        "bern-online (sequential static-trait), irt-mml (Gauss-Hermite logistic IRT), "
                        "irt-jml (StaticIRT joint MLE).")
    p.add_argument("--weights", choices=["irt", "free"], default="irt",
                   help="Bernstein weight parameterization: IRT-derived or freely learned (default: irt).")
    p.add_argument("--degree", type=int, default=10, help="Bernstein polynomial degree (default: 10).")
    p.add_argument("--num-nodes", type=int, default=None,
                   help="Quadrature nodes (default: 30 Gauss-Jacobi for bern-*, 21 Gauss-Hermite for irt-mml).")
    p.add_argument("--irt-model", choices=["1PL", "2PL", "3PL", "4PL"], default="2PL",
                   help="IRT family for irt-* models and IRT-parameterized weights (default: 2PL).")
    p.add_argument("--item-weight-rank", type=int, default=-1,
                   help="Approximation rank of the item weights (default: -1, full rank).")
    p.add_argument("--prior-alpha", type=float, default=1.0, help="Beta prior alpha (bern-*; default: 1.0).")
    p.add_argument("--prior-beta", type=float, default=1.0, help="Beta prior beta (bern-*; default: 1.0).")
    p.add_argument("--prior-mean", type=float, default=0.0, help="Normal prior mean (irt-mml; default: 0.0).")
    p.add_argument("--prior-std", type=float, default=1.0, help="Normal prior std (irt-mml; default: 1.0).")
    p.add_argument("--theta-reg", type=float, default=1.0, help="MAP strength on abilities (irt-jml; default: 1.0).")
    p.add_argument("--c-init", type=float, default=0.1, help="Initial guessing prob, 3PL/4PL (irt-jml; default: 0.1).")
    p.add_argument("--s-init", type=float, default=0.1, help="Initial slipping prob, 4PL (irt-jml; default: 0.1).")
    p.add_argument("--eap-num-nodes", type=int, default=40,
                   help="Gauss-Hermite nodes for irt-jml held-out EAP scoring (default: 40).")

    # Split / training / logging
    p.add_argument("--split", choices=["within", "within_random", "between"], default="within",
                   help="within (temporal tail), within_random (random item holdout), or between "
                        "(cold-start user holdout) (default: within).")
    p.add_argument("--train-val-test", default="0.7,0.1,0.2",
                   help="Comma-separated train,val,test proportions (default: 0.7,0.1,0.2).")
    p.add_argument("--lr", type=float, default=0.02, help="Learning rate (default: 0.02).")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    p.add_argument("--max-epochs", type=int, default=100, help="Max training epochs (default: 100).")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0).")
    p.add_argument("--patience", type=int, default=10, help="Early-stopping patience (default: 10).")
    p.add_argument("--accelerator", default="cpu", help="Lightning accelerator (default: cpu).")
    p.add_argument("--log-dir", default="lightning_logs", help="Metrics log dir (default: lightning_logs).")
    p.add_argument("--logger", choices=["csv", "tensorboard"], default="csv",
                   help="Metrics logger; tensorboard needs the `tensorboard` package (default: csv).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--out", default=None, help="Run dir (default: results/<dataset>/<timestamp>).")
    p.add_argument("--no-test", action="store_true", help="Skip the test phase after training.")
    p.add_argument("--no-predict", action="store_true", help="Skip writing predictions.csv.")


def _load_frame(args):
    """Resolve --data/--irw into (canonical response-log DataFrame, dataset name)."""
    if args.irw:
        path, name = ingest.resolve_irw(args.irw), args.irw
    else:
        path, name = args.data, Path(args.data).stem
    return ingest.load_table(path, skills=args.skills), name


def _build_task(args, stats):
    """Build (model, task, description) for the requested --model."""
    common = dict(num_users=stats.num_users, num_skills=stats.num_skills, num_items=stats.num_items,
                  item_weight_rank=args.item_weight_rank)

    if args.model.startswith("bern-"):
        from betabern.bernstein.estimator.exact import (
            build_exact_bernstein_estimator_free,
            build_exact_bernstein_estimator_irt,
        )
        from betabern.bernstein.estimator.online_quad import (
            build_online_quad_estimator_free,
            build_online_quad_estimator_irt,
        )
        from betabern.bernstein.estimator.quad import (
            build_quad_bernstein_estimator_free,
            build_quad_bernstein_estimator_irt,
        )
        from betabern.bernstein.task import OnlineQuadTask

        builders = {
            ("bern-exact", "irt"): build_exact_bernstein_estimator_irt,
            ("bern-exact", "free"): build_exact_bernstein_estimator_free,
            ("bern-quad", "irt"): build_quad_bernstein_estimator_irt,
            ("bern-quad", "free"): build_quad_bernstein_estimator_free,
            ("bern-online", "irt"): build_online_quad_estimator_irt,
            ("bern-online", "free"): build_online_quad_estimator_free,
        }
        common.update(degree_n=args.degree, prior=(args.prior_alpha, args.prior_beta))
        if args.weights == "irt":
            common["irt_model"] = args.irt_model
        if args.model != "bern-exact":
            common["num_nodes"] = args.num_nodes or 30
        model = builders[(args.model, args.weights)](**common)
        task_cls = OnlineQuadTask if args.model == "bern-online" else BayesianEstimatorTask
        task = task_cls(model, learning_rate=args.lr)
        tail = "exact" if args.model == "bern-exact" else f"Q={common['num_nodes']}"
        desc = f"{args.model}({args.weights}, degree={args.degree}, {tail})"

    elif args.model == "irt-mml":
        from betabern.irt.estimator.quad_irt import build_quad_irt_estimator
        num_nodes = args.num_nodes or 21
        model = build_quad_irt_estimator(**common, num_nodes=num_nodes, irt_model=args.irt_model,
                                         prior_mean=args.prior_mean, prior_std=args.prior_std)
        task = BayesianEstimatorTask(model, learning_rate=args.lr)
        desc = f"irt-mml({args.irt_model}, Q={num_nodes})"

    else:  # irt-jml
        from betabern.irt.estimator.static_irt import StaticIRT
        from betabern.irt.task import StaticIRTTask
        model = StaticIRT(**common, irt_model=args.irt_model, c_init=args.c_init, s_init=args.s_init)
        task = StaticIRTTask(model, learning_rate=args.lr, theta_reg=args.theta_reg,
                             eap_num_nodes=args.eap_num_nodes)
        desc = f"irt-jml({args.irt_model})"

    return task, desc


def main(args) -> None:
    L.seed_everything(args.seed)
    df, name = _load_frame(args)

    run_dir = args.out or os.path.join("results", name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args) | {"command": "fit"}, f, indent=4, default=str)

    ratios = tuple(map(float, args.train_val_test.split(",")))
    dm = harness.build_datamodule(df, split=args.split, train_val_test=ratios,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
    task, desc = _build_task(args, dm.stats)
    print(f"Model: {desc}")

    trainer = harness.fit(task, dm, max_epochs=args.max_epochs, accelerator=args.accelerator,
                          patience=args.patience, log_dir=args.log_dir, exp_name=args.model,
                          logger=args.logger, test=not args.no_test)
    if not args.no_predict:
        pred = harness.predict_frame(trainer, task, dm)
        out_path = os.path.join(run_dir, "predictions.csv")
        pred.to_csv(out_path, index=False)
        print(f"Predictions saved to {out_path}  ({len(pred)} rows)")
