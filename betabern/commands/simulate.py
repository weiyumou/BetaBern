"""``betabern simulate`` — generate a simulated IRT response log as a CSV.

The log carries the ground-truth ``user_theta`` / ``p_true_correct`` columns, which enable the
benchmark's recovery metrics; strip them with ``--no-truth`` to mimic a real dataset.

    betabern simulate --out data/simulated_student_log.csv --num-students 100 --num-items 20
"""
import argparse
from pathlib import Path

from betabern.core.data.simulate_irt import generate_log_data


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--num-students", type=int, default=100, help="Number of students (default: 100).")
    p.add_argument("--num-items", type=int, default=20, help="Items per skill (default: 20).")
    p.add_argument("--num-skills", type=int, default=5, help="Number of skills (default: 5).")
    p.add_argument("--min-interactions", type=int, default=80,
                   help="Min interactions per student-skill (default: 80).")
    p.add_argument("--max-interactions", type=int, default=100,
                   help="Max interactions per student-skill (default: 100).")
    p.add_argument("--drift-scale", type=float, default=0.0,
                   help="Ability drift per step; 0 = static trait (default: 0.0).")
    p.add_argument("--irt-model", choices=["1PL", "2PL", "3PL", "4PL"], default="4PL",
                   help="Generating IRT family (default: 4PL).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--no-truth", action="store_true",
                   help="Drop the ground-truth user_theta / p_true_correct columns.")
    p.add_argument("--items-out", default=None,
                   help="Optional CSV path for the ground-truth item-parameter table.")


def main(args) -> None:
    log_df, item_df = generate_log_data(
        args.num_students, args.num_items, args.min_interactions, args.max_interactions,
        n_skills=args.num_skills, drift_scale=args.drift_scale, irt_model=args.irt_model,
        random_state=args.seed)
    if args.no_truth:
        log_df = log_df.drop(columns=["user_theta", "p_true_correct"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(out, index=False)
    print(f"Wrote {len(log_df)} interactions ({args.num_students} students x {args.num_skills} skills) to {out}")
    if args.items_out:
        item_df.to_csv(args.items_out, index=False)
        print(f"Wrote ground-truth item parameters to {args.items_out}")
