"""
Simulated IRT response data with ground truth.

``generate_log_data`` returns an interaction log (``user_id, skill_name, item_id, is_correct,
timestamp``) plus two ground-truth columns — ``user_theta`` (the latent ability used to generate each
response) and ``p_true_correct`` (the generative P(correct)) — and a separate item-parameter table
(a/b/c/s). With ``drift_scale=0.0`` the ability is **static** (traditional IRT), which is what the
measurement benchmark uses for θ/probability recovery.
"""
import datetime

import numpy as np
import pandas as pd
from scipy.special import expit


def create_students(n_students, n_skills, random_state=None):
    """Generate students, each with a static latent ability per skill ~ N(0, 1)."""
    rng = np.random.default_rng(random_state)
    student_ids = [f"user-{i}" for i in range(n_students)]
    thetas = rng.normal(loc=0.0, scale=1.0, size=(n_students, n_skills))
    return pd.DataFrame({"student_id": student_ids, "thetas": list(thetas)})


def create_items(n_items, n_skills, irt_model: str = "4PL", random_state=None):
    """Generate item parameters via hierarchical (per-skill) sampling for the chosen IRT model."""
    rng = np.random.default_rng(random_state)

    irt_model = irt_model.upper()
    if irt_model not in ("1PL", "2PL", "3PL", "4PL"):
        raise ValueError(f"irt_model must be one of '1PL', '2PL', '3PL', '4PL', got '{irt_model}'")
    if n_items < n_skills:
        raise ValueError(
            f"Number of items ({n_items}) must be at least number of skills ({n_skills}) to ensure coverage.")

    item_ids = [f"item-{i}" for i in range(n_items)]

    # Assign skills: ensure each skill appears at least once, fill the rest randomly, then shuffle.
    base_skills = list(range(n_skills))
    remaining_skills = list(rng.integers(0, n_skills, size=n_items - n_skills))
    skill_indices = base_skills + remaining_skills
    rng.shuffle(skill_indices)
    skill_ids = [f"skill-{i}" for i in skill_indices]

    # Level 1: per-skill hyperparameters.
    base_dirichlet_alpha = np.array([2.0, 1.0, 7.0])  # priors for c, s, (1 - c - s)
    skill_configs = {
        s_id: {
            "mu_b": rng.normal(0.0, 1.0),
            "sigma_b": rng.gamma(shape=2.0, scale=0.25),
            "mu_log_a": rng.normal(0.0, 0.3),
            "sigma_log_a": rng.gamma(shape=2.0, scale=0.1),
            "dirichlet_alpha": base_dirichlet_alpha * rng.uniform(0.5, 1.5),
        }
        for s_id in range(n_skills)
    }

    # Level 2: per-item parameters drawn from their skill's distribution.
    a_params, b_params = np.zeros(n_items), np.zeros(n_items)
    c_params, s_params = np.zeros(n_items), np.zeros(n_items)
    for i in range(n_items):
        config = skill_configs[skill_indices[i]]
        b_params[i] = rng.normal(config["mu_b"], config["sigma_b"])
        a_params[i] = 1.0 if irt_model == "1PL" else rng.lognormal(config["mu_log_a"], config["sigma_log_a"])
        if irt_model in ("1PL", "2PL"):
            c_params[i], s_params[i] = 0.0, 0.0
        elif irt_model == "3PL":
            c_params[i], s_params[i] = rng.dirichlet(config["dirichlet_alpha"])[0], 0.0
        else:  # 4PL
            simplex = rng.dirichlet(config["dirichlet_alpha"])
            c_params[i], s_params[i] = simplex[0], simplex[1]

    return pd.DataFrame({
        "item_id": item_ids, "skill_id": skill_ids,
        "a_irt": a_params, "b_irt": b_params, "c_guess": c_params, "s_slip": s_params,
    })


def get_ground_truth_prob(theta, a_item, b_item, c_item, s_item):
    """Standard 4PL IRT: P(theta) = c + (1 - s - c) * sigmoid(a * (theta - b))."""
    p_correct = c_item + (1.0 - s_item - c_item) * expit(a_item * (theta - b_item))
    return np.clip(p_correct, 0.0, 1.0)


def generate_dynamic_data(student_df, item_df, interactions_per_student_skill,
                          drift_scale: float = 0.0, random_state=None):
    """Generate an interaction log; ``drift_scale=0`` keeps each student's ability static over time."""
    rng = np.random.default_rng(random_state)
    log_data = []

    n_skills = len(student_df.iloc[0]["thetas"])
    items_by_skill = {s: item_df[item_df["skill_id"] == f"skill-{s}"] for s in range(n_skills)}
    base_time = datetime.datetime(2025, 10, 1, 10, 30, 0).timestamp()

    for student_idx, student in student_df.iterrows():
        current_thetas = np.array(student["thetas"], dtype=float)

        student_interactions = []
        for skill_idx in range(n_skills):
            n_inter = interactions_per_student_skill[student_idx, skill_idx]
            skill_items = items_by_skill[skill_idx]
            if n_inter == 0 or len(skill_items) == 0:
                continue
            for item_idx in rng.choice(len(skill_items), size=n_inter, replace=True):
                student_interactions.append((skill_idx, skill_items.iloc[item_idx]))
        rng.shuffle(student_interactions)

        n_total = len(student_interactions)
        if n_total == 0:
            continue
        student_offset = rng.integers(0, 60 * 60 * 24 * 10)  # within 10 days
        timestamps = base_time + student_offset + np.cumsum(rng.exponential(1, size=n_total)) * 60 * 60

        for k, (skill_idx, item_params) in enumerate(student_interactions):
            # Drift step (sample always to keep the RNG stream consistent across drift settings).
            potential_drift = rng.normal(0.0, drift_scale if drift_scale > 0 else 1.0)
            current_thetas[skill_idx] += potential_drift if drift_scale > 0 else 0.0
            theta = current_thetas[skill_idx]

            p_correct = get_ground_truth_prob(
                theta, item_params["a_irt"], item_params["b_irt"], item_params["c_guess"], item_params["s_slip"])
            log_data.append({
                "user_id": student["student_id"],
                "skill_name": item_params["skill_id"],
                "item_id": item_params["item_id"],
                "is_correct": int(rng.binomial(1, p_correct)),
                "timestamp": int(timestamps[k]),
                "p_true_correct": p_correct,
                "user_theta": theta,
            })

    df = pd.DataFrame(log_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    return df


def generate_log_data(num_students, num_items, min_interactions, max_interactions,
                      n_skills=1, drift_scale=0.0, irt_model="4PL", random_state=None):
    """Generate simulated IRT log data.

    :return: ``(log_df, item_df)`` — the interaction log (with ``user_theta`` / ``p_true_correct``
        ground-truth columns) and the ground-truth item-parameter table.
    """
    student_seed, item_seed, data_seed = np.random.SeedSequence(random_state).spawn(3)
    student_df = create_students(num_students, n_skills, random_state=student_seed)
    item_df = create_items(num_items, n_skills, irt_model=irt_model, random_state=item_seed)

    data_rng = np.random.default_rng(data_seed)
    interactions = data_rng.integers(min_interactions, max_interactions + 1, size=(num_students, n_skills))
    log_df = generate_dynamic_data(student_df, item_df, interactions,
                                   drift_scale=drift_scale, random_state=data_rng)
    return log_df, item_df


if __name__ == "__main__":
    N_STUDENTS, N_ITEMS, N_SKILLS = 100, 20, 5
    MIN_INTERACTIONS, MAX_INTERACTIONS = 80, 100

    log_df, item_df = generate_log_data(N_STUDENTS, N_ITEMS, MIN_INTERACTIONS, MAX_INTERACTIONS,
                                        n_skills=N_SKILLS, drift_scale=0.05, random_state=42)

    print(f"--- Generated {len(log_df)} interactions ---")
    print(log_df.head(10))
    print("\nGround truth item parameters:")
    print(item_df.head(10))
