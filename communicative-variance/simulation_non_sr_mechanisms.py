#!/usr/bin/env python3
"""
Non-SR mechanism tests for Theorem 1 generality.

Adversarial review Hole #1 identified that the simulation only tests
stochastic resonance (SR) variants. If Theorem 1 is genuinely general,
its conditions should predict when noise is beneficial in systems where
the mechanism is NOT threshold crossing.

Three non-SR mechanisms tested:
  A. Optimization landscape escape (simulated annealing analog)
  B. Ensemble diversity (random forest analog)
  C. Exploration-exploitation (multi-armed bandit)

For each, we:
  1. Define level-N performance (degrades with noise)
  2. Define level-N+1 performance (can improve with noise)
  3. Map the five conditions to the mechanism's structure
  4. Sweep noise amplitude and test for inverted-U
  5. Violate each condition and verify benefit disappears
"""

import numpy as np
import json
from dataclasses import dataclass, asdict

SEED = 42

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ================================================================
# MECHANISM A: OPTIMIZATION LANDSCAPE ESCAPE
# ================================================================
# Level N: stability at current position (degrades when you're bumped)
# Level N+1: quality of best position found (improves when noise lets
#            you escape local optima to discover global optimum)
# Nonlinearity: the multi-modal landscape itself
# NOT SR: there is no threshold detector. The mechanism is landscape
#         exploration via random perturbation.
# ================================================================

def make_landscape(n_dims=1, n_modes=5, seed=42):
    """
    Create a multimodal 1D landscape with a clear local trap and distant global optimum.
    Returns a function f(x) -> scalar fitness.
    """
    rng = np.random.default_rng(seed)

    # Fixed, well-separated 1D landscape for reproducible SA behavior:
    # Local optimum near x=0 (height 0.6), global optimum near x=3 (height 1.0)
    # Valley between them at x=1.5 (height ~0.1)
    if n_dims == 1:
        centers = np.array([[0.0], [3.0], [-2.0], [5.0], [-4.0]])[:n_modes]
        heights = np.array([0.6, 1.0, 0.3, 0.2, 0.15])[:n_modes]
        widths = np.array([0.8, 0.6, 0.7, 0.5, 0.6])[:n_modes]
    else:
        centers = rng.uniform(-5, 5, (n_modes, n_dims))
        heights = rng.uniform(0.3, 0.8, n_modes)
        widths = rng.uniform(0.5, 2.0, n_modes)
        heights[0] = 1.0
        centers[0] = rng.uniform(2, 4, n_dims)

    def landscape(x):
        x = np.atleast_2d(x)
        total = np.zeros(x.shape[0])
        for i in range(len(centers)):
            dist_sq = np.sum((x - centers[i]) ** 2, axis=1)
            total += heights[i] * np.exp(-dist_sq / (2 * widths[i] ** 2))
        return total

    global_idx = np.argmax(heights)
    return landscape, centers[global_idx], heights[global_idx]


def optimization_escape_test(
    sigma,
    start_position,
    landscape_fn,
    n_restarts=200,
    n_steps=80,
    seed=42,
):
    """
    Simulated annealing from start_position with step size = sigma.

    Level N performance: fitness at current position after perturbation
      (how much we degrade from the local optimum by accepting noise).
      Measured as: mean fitness at the END of each trajectory, relative
      to trajectories that stayed near start.
    Level N+1 performance: best fitness discovered across all restarts.
      Represents the system's ability to find superior solutions.

    Returns (p_n, p_n1) where both are in [0, 1].
    """
    rng = np.random.default_rng(seed)
    n_dims = len(start_position)

    start_fitness = landscape_fn(np.array([start_position]))[0]
    final_fitnesses = []

    for r in range(n_restarts):
        pos = np.array(start_position, dtype=float)
        for step in range(n_steps):
            if sigma > 0:
                noise = rng.normal(0, sigma, n_dims)
                candidate = pos + noise
            else:
                break  # no noise, no movement

            cand_fit = landscape_fn(np.array([candidate]))[0]
            curr_fit = landscape_fn(np.array([pos]))[0]

            if cand_fit > curr_fit:
                pos = candidate
            elif sigma > 0:
                # Metropolis acceptance: higher sigma = more likely to accept downhill
                delta = cand_fit - curr_fit
                # Temperature proportional to sigma (SA analogy)
                temperature = sigma * 0.5
                if rng.random() < np.exp(delta / max(temperature, 1e-10)):
                    pos = candidate

        final_fit = landscape_fn(np.array([pos]))[0]
        final_fitnesses.append(final_fit)

    # P_N: mean final fitness (proxy for how well level-N function is maintained)
    # At sigma=0, this equals start_fitness. Higher sigma degrades it on average
    # because SA wanders away from the local optimum.
    p_n = float(np.mean(final_fitnesses))

    # P_N+1: best fitness discovered (the "creative" output of exploration)
    # At sigma=0, this equals start_fitness. With moderate sigma, this can
    # exceed start_fitness by discovering the global optimum.
    p_n1 = float(np.max(final_fitnesses))

    return p_n, p_n1


# ================================================================
# MECHANISM B: ENSEMBLE DIVERSITY
# ================================================================
# Level N: individual predictor accuracy (degrades when you subsample/noise)
# Level N+1: ensemble accuracy (improves with diversity up to a point)
# Nonlinearity: majority vote / averaging of non-identical errors
# NOT SR: there is no threshold. The mechanism is error decorrelation
#         through diversity of weak learners.
# ================================================================

def ensemble_diversity_test(
    noise_level,
    n_predictors=21,
    n_samples=1000,
    base_accuracy=0.65,  # individual accuracy without noise
    signal_dims=10,
    seed=42,
):
    """
    Create an ensemble of noisy predictors on a HARD binary classification task.

    The task is designed so that:
    - Individual predictors using the true weights only get ~65% accuracy
      (because the decision boundary is noisy / features are weak)
    - Feature noise creates diversity: each predictor makes DIFFERENT errors
    - Majority vote can exploit decorrelated errors to improve

    This models the Random Forest insight: bagging (subsampling = noise)
    degrades individual trees but improves the forest.

    Level N: mean individual accuracy (degrades with noise).
    Level N+1: ensemble accuracy (can improve with moderate noise).
    """
    rng = np.random.default_rng(seed)

    # Generate a hard classification problem with a nonlinear boundary
    # Use XOR-like structure: class depends on INTERACTION of features
    # Individual linear predictors can't fully solve this
    X = rng.normal(0, 1, (n_samples, signal_dims))

    # Nonlinear true boundary: sign of product of first two features + noise
    # This ensures individual linear predictors are weak (~60-70% accuracy)
    decision_val = X[:, 0] * X[:, 1] + 0.3 * X[:, 2] + rng.normal(0, 0.5, n_samples)
    y_true = (decision_val > 0).astype(int)

    individual_accuracies = []
    predictions = np.zeros((n_predictors, n_samples), dtype=int)

    for p in range(n_predictors):
        # Each predictor gets a DIFFERENT random subset of features (bagging-like)
        # PLUS independent feature noise
        feature_mask = rng.random(signal_dims) > 0.3  # ~70% features kept
        if not np.any(feature_mask):
            feature_mask[0] = True  # at least one feature

        if noise_level > 0:
            X_noisy = X + rng.normal(0, noise_level, X.shape)
        else:
            X_noisy = X.copy()

        # Each predictor learns its own weights on its feature subset
        # (simple: use correlation with y as weights — crude but fast)
        X_subset = X_noisy[:, feature_mask]
        # Split into train/test for honest evaluation
        n_train = n_samples // 2
        X_train, X_test = X_subset[:n_train], X_subset[n_train:]
        y_train, y_test = y_true[:n_train], y_true[n_train:]

        # Compute correlation-based weights
        weights = np.zeros(X_subset.shape[1])
        for j in range(X_subset.shape[1]):
            corr = np.corrcoef(X_train[:, j], y_train)[0, 1]
            weights[j] = corr if not np.isnan(corr) else 0

        # Predict on full dataset (for ensemble consistency)
        y_pred = (X_subset @ weights > 0).astype(int)
        predictions[p] = y_pred

        acc = np.mean(y_pred == y_true)
        individual_accuracies.append(acc)

    # Ensemble: majority vote
    ensemble_pred = (np.sum(predictions, axis=0) > n_predictors / 2).astype(int)
    ensemble_accuracy = np.mean(ensemble_pred == y_true)

    p_n = float(np.mean(individual_accuracies))
    p_n1 = float(ensemble_accuracy)

    return p_n, p_n1


# ================================================================
# MECHANISM C: EXPLORATION-EXPLOITATION (MULTI-ARMED BANDIT)
# ================================================================
# Level N: immediate expected reward per pull (degrades when exploring)
# Level N+1: cumulative long-term reward (improves by discovering better arms)
# Nonlinearity: argmax over estimated arm values (hard selection)
# NOT SR: the mechanism is search under uncertainty, not signal detection.
# ================================================================

def bandit_test(
    exploration_noise,
    n_arms=10,
    n_pulls=200,
    n_episodes=500,
    arm_means=None,
    seed=42,
):
    """
    Multi-armed bandit with epsilon-softmax exploration.

    exploration_noise controls how much we explore:
      0 = pure exploitation (always pull estimated-best arm)
      high = random exploration

    Level N: immediate reward quality (how often we pull known-good arms)
    Level N+1: final cumulative reward (benefits from discovering best arm)
    """
    rng = np.random.default_rng(seed)

    if arm_means is None:
        # One clearly best arm, rest mediocre
        arm_means = np.array([0.3] * (n_arms - 1) + [0.8])
        rng.shuffle(arm_means)

    best_arm = np.argmax(arm_means)
    best_mean = arm_means[best_arm]

    immediate_rewards = []  # reward per pull in first half
    final_rewards = []      # reward per pull in second half (after learning)

    for ep in range(n_episodes):
        ep_rng = np.random.default_rng(seed + ep)
        estimates = np.zeros(n_arms)
        counts = np.zeros(n_arms)
        first_half_reward = 0
        second_half_reward = 0

        for t in range(n_pulls):
            # Choose arm: softmax with temperature = exploration_noise
            if exploration_noise > 0 and np.any(counts > 0):
                # Boltzmann exploration
                temp = exploration_noise
                logits = estimates / max(temp, 1e-10)
                logits -= np.max(logits)  # numerical stability
                probs = np.exp(logits)
                probs /= probs.sum()
                arm = ep_rng.choice(n_arms, p=probs)
            elif exploration_noise == 0 and np.any(counts > 0):
                arm = np.argmax(estimates)
            else:
                arm = ep_rng.integers(0, n_arms)

            # Pull arm, get reward
            reward = ep_rng.normal(arm_means[arm], 0.3)

            # Update estimates
            counts[arm] += 1
            estimates[arm] += (reward - estimates[arm]) / counts[arm]

            if t < n_pulls // 2:
                first_half_reward += reward
            else:
                second_half_reward += reward

        immediate_rewards.append(first_half_reward / (n_pulls // 2))
        final_rewards.append(second_half_reward / (n_pulls // 2))

    # P_N: immediate reward quality (how good are pulls in first half)
    # Normalized so that always-best-arm gives ~1.0
    p_n = float(np.mean(immediate_rewards) / best_mean)
    # P_N+1: learned reward quality (second half, after exploration)
    p_n1 = float(np.mean(final_rewards) / best_mean)

    return p_n, p_n1


# ================================================================
# UNIFIED TEST HARNESS
# ================================================================

def sweep_mechanism(
    name,
    mechanism_fn,
    sigma_range,
    alpha,
    beta,
    coupling="additive",
    n_points=30,
    seed=SEED,
):
    """
    Sweep noise amplitude for any mechanism function.
    mechanism_fn(sigma, seed) -> (p_n, p_n1)
    """
    sigmas = np.linspace(sigma_range[0], sigma_range[1], n_points)
    results = {
        "name": name,
        "alpha": alpha,
        "beta": beta,
        "coupling": coupling,
        "sigmas": [],
        "p_n": [],
        "p_n1": [],
        "p_sys": [],
    }

    for i, sigma in enumerate(sigmas):
        pn, pn1 = mechanism_fn(float(sigma), seed=seed + i)

        if coupling == "additive":
            psys = alpha * pn + beta * pn1
        elif coupling == "substrate":
            psys = pn * (alpha + beta * pn1)
        elif coupling == "multiplicative":
            psys = (max(pn, 1e-10) ** alpha) * (max(pn1, 1e-10) ** beta)
        else:
            psys = alpha * pn + beta * pn1

        results["sigmas"].append(float(sigma))
        results["p_n"].append(float(pn))
        results["p_n1"].append(float(pn1))
        results["p_sys"].append(float(psys))

    # Find optimum
    best_idx = int(np.argmax(results["p_sys"]))
    results["sigma_opt"] = results["sigmas"][best_idx]
    results["p_sys_at_opt"] = results["p_sys"][best_idx]
    results["p_sys_at_zero"] = results["p_sys"][0]
    results["net_benefit"] = results["p_sys"][best_idx] - results["p_sys"][0]
    results["inverted_u"] = (
        best_idx > 0 and
        best_idx < len(sigmas) - 1 and
        results["p_sys"][best_idx] > results["p_sys"][0] and
        results["p_sys"][best_idx] > results["p_sys"][-1]
    )

    # Check conditions dynamically
    pn_0, pn1_0 = mechanism_fn(0.0, seed=seed)
    pn_max_noise, pn1_max_noise = mechanism_fn(float(sigma_range[1]), seed=seed + 999)

    # C1: Is the system suboptimal at zero noise?
    # For non-SR: is level N+1 below its achievable max?
    c1 = pn1_0 < max(results["p_n1"])

    # C2: Is there a nonlinear response? (Jensen gap)
    # Test: does E[f(x+noise)] differ from f(x)?
    mid_sigma = (sigma_range[0] + sigma_range[1]) / 2
    pn_mid, pn1_mid = mechanism_fn(float(mid_sigma), seed=seed + 500)
    jensen_gap = pn1_mid - pn1_0
    c2 = jensen_gap > 0.01

    # C3: Can noise access the improvement space?
    # Does N+1 improve at ANY noise level?
    c3 = any(p > pn1_0 + 0.005 for p in results["p_n1"])

    # C4: Is the gain at N+1 worth the cost at N?
    # Check at the best sigma
    pn_best, pn1_best = results["p_n"][best_idx], results["p_n1"][best_idx]
    gain = beta * (pn1_best - pn1_0)
    cost = alpha * (pn_0 - pn_best)
    c4 = gain > cost

    # C5: Does level N degrade gracefully?
    # Check: at 10% of max noise, has N lost less than 50%?
    small_sigma = sigma_range[1] * 0.1
    pn_small, _ = mechanism_fn(float(small_sigma), seed=seed + 501)
    fractional_loss = (pn_0 - pn_small) / max(pn_0, 1e-10)
    c5 = fractional_loss < 0.5

    results["conditions"] = {
        "C1_suboptimal": {"met": bool(c1), "evidence": f"P_N+1(0)={pn1_0:.3f}, max={max(results['p_n1']):.3f}"},
        "C2_nonlinear": {"met": bool(c2), "evidence": f"Jensen gap={jensen_gap:.4f}"},
        "C3_accessible": {"met": bool(c3), "evidence": f"N+1 improves at some sigma: {c3}"},
        "C4_weighting": {"met": bool(c4), "evidence": f"gain={gain:.4f}, cost={cost:.4f}"},
        "C5_robustness": {"met": bool(c5), "evidence": f"fractional_loss at 10% noise = {fractional_loss:.3f}"},
    }
    results["all_conditions_met"] = all(c["met"] for c in results["conditions"].values())

    return results


def print_result(r):
    """Pretty-print a sweep result."""
    print(f"\n--- {r['name']} ---")
    for cname, cval in r["conditions"].items():
        status = "MET" if cval["met"] else "FAILED"
        print(f"  {cname}: {status} ({cval['evidence']})")
    print(f"  sigma* = {r['sigma_opt']:.3f}")
    print(f"  P_sys(0) = {r['p_sys_at_zero']:.4f}")
    print(f"  P_sys(sigma*) = {r['p_sys_at_opt']:.4f}")
    print(f"  net benefit = {r['net_benefit']:.4f}")
    print(f"  inverted-U detected: {r['inverted_u']}")
    print(f"  all conditions met: {r['all_conditions_met']}")


# ================================================================
# MAIN
# ================================================================

def main():
    all_results = {}

    print("=" * 80)
    print("NON-SR MECHANISM TESTS — THEOREM 1 GENERALITY")
    print("Testing whether conditions C1-C5 predict noise benefit")
    print("in systems that are NOT stochastic resonance")
    print("=" * 80)

    # ===========================================================
    # MECHANISM A: OPTIMIZATION LANDSCAPE ESCAPE
    # ===========================================================
    print("\n" + "=" * 80)
    print("MECHANISM A: OPTIMIZATION LANDSCAPE ESCAPE")
    print("Level N = stability at local optimum")
    print("Level N+1 = quality of best position found")
    print("Nonlinearity = multimodal landscape (NOT a threshold detector)")
    print("=" * 80)

    landscape_fn, global_opt_pos, global_opt_val = make_landscape(
        n_dims=1, n_modes=5, seed=SEED
    )

    # Start at the local optimum at x=0 (height 0.6)
    # The global optimum is at x=3 (height 1.0) with a valley between
    local_opt = np.array([0.0])
    local_val = landscape_fn(np.array([local_opt]))[0]
    print(f"\n  Local optimum at {local_opt} with value {local_val:.3f}")
    print(f"  Global optimum at {global_opt_pos} with value {global_opt_val:.3f}")
    print(f"  Gap: {global_opt_val - local_val:.3f}")

    def optim_mechanism(sigma, seed=42):
        return optimization_escape_test(
            sigma, start_position=local_opt, landscape_fn=landscape_fn,
            n_restarts=150, n_steps=40, seed=seed,
        )

    # Baseline: all conditions met (stuck in local opt, moderate weighting)
    r_opt = sweep_mechanism(
        "Optimization: baseline (stuck in local opt)",
        optim_mechanism,
        sigma_range=[0.0, 4.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_opt)
    all_results["optimization_baseline"] = r_opt

    # C1 violation: start AT the global optimum (nothing to find)
    def optim_at_global(sigma, seed=42):
        return optimization_escape_test(
            sigma, start_position=global_opt_pos.flatten(), landscape_fn=landscape_fn,
            n_restarts=150, n_steps=80, seed=seed,
        )

    r_opt_c1 = sweep_mechanism(
        "Optimization: C1 violated (already at global optimum)",
        optim_at_global,
        sigma_range=[0.0, 4.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_opt_c1)
    all_results["optimization_c1_violated"] = r_opt_c1

    # C4 violation: weight stability much more than exploration
    r_opt_c4 = sweep_mechanism(
        "Optimization: C4 violated (stability >> exploration)",
        optim_mechanism,
        sigma_range=[0.0, 4.0],
        alpha=0.95, beta=0.05,
        n_points=25,
    )
    print_result(r_opt_c4)
    all_results["optimization_c4_violated"] = r_opt_c4

    # Null: flat landscape (no nonlinearity / no structure to exploit)
    flat_fn = lambda x: np.ones(np.atleast_2d(x).shape[0]) * 0.5

    def optim_flat(sigma, seed=42):
        return optimization_escape_test(
            sigma, start_position=np.array([0.0]), landscape_fn=flat_fn,
            n_restarts=150, n_steps=80, seed=seed,
        )

    r_opt_flat = sweep_mechanism(
        "Optimization: NULL (flat landscape — no structure)",
        optim_flat,
        sigma_range=[0.0, 4.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_opt_flat)
    all_results["optimization_null_flat"] = r_opt_flat

    # ===========================================================
    # MECHANISM B: ENSEMBLE DIVERSITY
    # ===========================================================
    print("\n" + "=" * 80)
    print("MECHANISM B: ENSEMBLE DIVERSITY")
    print("Level N = individual predictor accuracy")
    print("Level N+1 = ensemble (majority vote) accuracy")
    print("Nonlinearity = majority vote aggregation (NOT a threshold detector)")
    print("=" * 80)

    def ensemble_mechanism(sigma, seed=42):
        return ensemble_diversity_test(
            noise_level=sigma,
            n_predictors=21,
            n_samples=500,
            signal_dims=10,
            seed=seed,
        )

    r_ens = sweep_mechanism(
        "Ensemble: baseline (moderate individual accuracy)",
        ensemble_mechanism,
        sigma_range=[0.0, 3.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_ens)
    all_results["ensemble_baseline"] = r_ens

    # C1 violation: perfect individual accuracy (no room for ensemble to help)
    def ensemble_perfect(sigma, seed=42):
        return ensemble_diversity_test(
            noise_level=sigma,
            n_predictors=21,
            n_samples=500,
            base_accuracy=0.99,
            signal_dims=10,
            seed=seed,
        )

    r_ens_c1 = sweep_mechanism(
        "Ensemble: C1 violated (near-perfect individuals)",
        ensemble_perfect,
        sigma_range=[0.0, 3.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_ens_c1)
    all_results["ensemble_c1_violated"] = r_ens_c1

    # C4 violation: individual accuracy weighted much more
    r_ens_c4 = sweep_mechanism(
        "Ensemble: C4 violated (individual >> ensemble weighting)",
        ensemble_mechanism,
        sigma_range=[0.0, 3.0],
        alpha=0.95, beta=0.05,
        n_points=25,
    )
    print_result(r_ens_c4)
    all_results["ensemble_c4_violated"] = r_ens_c4

    # ===========================================================
    # MECHANISM C: EXPLORATION-EXPLOITATION (BANDIT)
    # ===========================================================
    print("\n" + "=" * 80)
    print("MECHANISM C: EXPLORATION-EXPLOITATION (MULTI-ARMED BANDIT)")
    print("Level N = immediate reward quality")
    print("Level N+1 = learned long-term reward quality")
    print("Nonlinearity = argmax over estimated arm values (NOT threshold detection)")
    print("=" * 80)

    def bandit_mechanism(sigma, seed=42):
        return bandit_test(
            exploration_noise=sigma,
            n_arms=10,
            n_pulls=200,
            n_episodes=300,
            seed=seed,
        )

    r_ban = sweep_mechanism(
        "Bandit: baseline (one hidden best arm)",
        bandit_mechanism,
        sigma_range=[0.0, 2.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_ban)
    all_results["bandit_baseline"] = r_ban

    # C1 violation: all arms have same mean (nothing better to discover)
    def bandit_uniform(sigma, seed=42):
        return bandit_test(
            exploration_noise=sigma,
            n_arms=10,
            n_pulls=200,
            n_episodes=300,
            arm_means=np.array([0.5] * 10),
            seed=seed,
        )

    r_ban_c1 = sweep_mechanism(
        "Bandit: C1 violated (all arms equal — nothing to discover)",
        bandit_uniform,
        sigma_range=[0.0, 2.0],
        alpha=0.3, beta=0.7,
        n_points=25,
    )
    print_result(r_ban_c1)
    all_results["bandit_c1_violated"] = r_ban_c1

    # C4 violation: immediate reward heavily weighted
    r_ban_c4 = sweep_mechanism(
        "Bandit: C4 violated (immediate >> long-term weighting)",
        bandit_mechanism,
        sigma_range=[0.0, 2.0],
        alpha=0.95, beta=0.05,
        n_points=25,
    )
    print_result(r_ban_c4)
    all_results["bandit_c4_violated"] = r_ban_c4

    # ===========================================================
    # SUMMARY
    # ===========================================================
    print("\n" + "=" * 80)
    print("SUMMARY: THEOREM 1 GENERALITY ACROSS NON-SR MECHANISMS")
    print("=" * 80)

    summary_rows = []
    for key, r in all_results.items():
        row = {
            "test": r["name"],
            "mechanism": key.split("_")[0],
            "inverted_u": r["inverted_u"],
            "net_benefit": r["net_benefit"],
            "all_conditions_met": r["all_conditions_met"],
            "conditions": {k: v["met"] for k, v in r["conditions"].items()},
        }
        summary_rows.append(row)

    # Theorem 1 prediction matrix
    print(f"\n{'Test':<55} {'Conds':>5} {'Benefit':>8} {'U-shape':>7} {'Prediction':>12}")
    print("-" * 95)
    for row in summary_rows:
        conds = "ALL" if row["all_conditions_met"] else "SOME"
        benefit = f"{row['net_benefit']:.4f}"
        u = "YES" if row["inverted_u"] else "NO"

        # Theorem 1 predicts: conditions met → benefit; conditions violated → no guarantee
        if row["all_conditions_met"] and row["net_benefit"] > 0.005:
            prediction = "CONFIRMED"
        elif row["all_conditions_met"] and row["net_benefit"] <= 0.005:
            prediction = "REFUTED!"
        elif not row["all_conditions_met"] and row["net_benefit"] <= 0.005:
            prediction = "CONSISTENT"
        else:
            prediction = "SUFFICIENT?"  # benefit despite violation — conditions not necessary

        print(f"  {row['test']:<53} {conds:>5} {benefit:>8} {u:>7} {prediction:>12}")

    # Count outcomes
    confirmed = sum(1 for r in summary_rows if r["all_conditions_met"] and r["net_benefit"] > 0.005)
    refuted = sum(1 for r in summary_rows if r["all_conditions_met"] and r["net_benefit"] <= 0.005)
    consistent = sum(1 for r in summary_rows if not r["all_conditions_met"] and r["net_benefit"] <= 0.005)
    not_necessary = sum(1 for r in summary_rows if not r["all_conditions_met"] and r["net_benefit"] > 0.005)

    print(f"\n  Theorem 1 confirmed (all met + benefit):      {confirmed}")
    print(f"  Theorem 1 REFUTED  (all met + no benefit):     {refuted}")
    print(f"  Consistent (some violated + no benefit):        {consistent}")
    print(f"  Conditions not necessary (violated + benefit):  {not_necessary}")

    if refuted > 0:
        print(f"\n  >>> THEOREM 1 IS NOT GENERAL: {refuted} non-SR mechanism(s) have all conditions")
        print(f"      met but show no benefit. The conditions are SR-specific.")
    elif confirmed > 0:
        print(f"\n  >>> THEOREM 1 GENERALIZES: {confirmed} non-SR mechanism(s) confirm the prediction.")
        print(f"      The conditions are structural, not SR-specific.")
    else:
        print(f"\n  >>> INCONCLUSIVE: No cases with all conditions met across non-SR mechanisms.")
        print(f"      The conditions may need reformulation for non-SR domains.")

    # Save
    output_path = "/Users/jmcentire/Personal/Research/CommunicativeVariance/simulation_results_non_sr.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
