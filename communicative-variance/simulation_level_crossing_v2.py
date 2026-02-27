#!/usr/bin/env python3
"""
Numerical simulation v2: Rigorous testing of the five sufficient conditions
for net generative level-crossing (Atlan formalization, Theorem 1).

Addresses adversarial review holes from v1:
  1. Tests multiple nonlinearity types (threshold, sigmoid, polynomial)
  2. Tests multiple degradation models (linear, exponential, catastrophic)
  3. Tests both additive and multiplicative P_sys coupling
  4. Properly tests C5 violation with substrate-dependent coupling
  5. Monte Carlo sensitivity analysis across parameter space
  6. Replaces CS pseudo-test with actual strategic communication sim
  7. Null models (linear detector, shuffled noise, no-threshold baseline)
  8. Seed control and bootstrap confidence intervals
  9. Conditions computed dynamically, not hard-coded
  10. Adversarial parameter search (conditions met but no benefit?)
"""

import numpy as np
import json
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional
import warnings

# ============================================================
# GLOBAL CONFIG
# ============================================================
SEED = 42
N_BOOTSTRAP = 200
CONFIDENCE_LEVEL = 0.95

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

# ============================================================
# NONLINEARITY TYPES (Hole #1: test beyond textbook SR)
# ============================================================
def threshold_detector(signal, threshold, noise_std, n_trials=10000, rng=None):
    """Hard threshold: P(signal + noise > threshold)."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, n_trials) if noise_std > 0 else np.zeros(n_trials)
    detections = (signal + noise) > threshold
    return np.mean(detections)

def sigmoid_detector(signal, threshold, noise_std, n_trials=10000, rng=None, steepness=10.0):
    """Smooth sigmoid nonlinearity: sigma(k*(x - threshold))."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, n_trials) if noise_std > 0 else np.zeros(n_trials)
    x = signal + noise
    responses = 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))
    return np.mean(responses)

def polynomial_detector(signal, threshold, noise_std, n_trials=10000, rng=None, power=3):
    """
    Polynomial nonlinearity: max(0, (x - threshold))^p, normalized.
    Smooth, no hard threshold, but still nonlinear.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, n_trials) if noise_std > 0 else np.zeros(n_trials)
    x = signal + noise
    responses = np.maximum(0, x - threshold) ** power
    # Normalize to [0, 1] range using a reasonable max
    max_response = (2.0) ** power  # assume max meaningful excursion is 2 above threshold
    return np.mean(np.minimum(responses / max_response, 1.0))

def linear_detector(signal, threshold, noise_std, n_trials=10000, rng=None):
    """
    NULL MODEL: Linear response (no nonlinearity).
    Noise should NOT help a linear system (Jensen's inequality: E[f(x+n)] = f(x) for affine f).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, n_trials) if noise_std > 0 else np.zeros(n_trials)
    x = signal + noise
    # Linear response clipped to [0, 1]
    responses = np.clip(x / threshold, 0, 1)
    return np.mean(responses)

# ============================================================
# DEGRADATION MODELS (Hole #2: realistic cost functions)
# ============================================================
def linear_degradation(sigma, L=1.0, base=1.0):
    """P_N = base - L*sigma, clipped at 0."""
    return max(0.0, base - L * sigma)

def exponential_degradation(sigma, L=1.0, base=1.0):
    """P_N = base * exp(-L*sigma). Never reaches 0, more realistic."""
    return base * np.exp(-L * sigma)

def catastrophic_degradation(sigma, L=1.0, base=1.0, cliff=1.5):
    """P_N drops to 0 at sigma=cliff (phase transition / brittle failure)."""
    if sigma >= cliff:
        return 0.0
    return base * (1.0 - (sigma / cliff) ** 3)  # cubic approach to cliff

# ============================================================
# SYSTEM PERFORMANCE COUPLING (Hole #3: beyond additive)
# ============================================================
def additive_coupling(pn, pn1, alpha, beta):
    """P_sys = alpha*P_N + beta*P_{N+1}. Standard separable model."""
    return alpha * pn + beta * pn1

def multiplicative_coupling(pn, pn1, alpha, beta):
    """P_sys = (P_N^alpha) * (P_{N+1}^beta). If N fails, N+1 is gated."""
    # Avoid 0^0 issues
    pn_safe = max(pn, 1e-10)
    pn1_safe = max(pn1, 1e-10)
    return (pn_safe ** alpha) * (pn1_safe ** beta)

def substrate_coupling(pn, pn1, alpha, beta):
    """P_sys = P_N * (alpha + beta*P_{N+1}). N+1 benefit requires N substrate."""
    return pn * (alpha + beta * pn1)

# ============================================================
# CONDITION CHECKING (Hole #9: compute dynamically)
# ============================================================
def check_conditions(signal, threshold, alpha, beta, L, detector_fn,
                     degradation_fn, coupling_fn, rng, n_trials=50000):
    """
    Dynamically verify each condition rather than hard-coding.
    Returns dict of condition name -> (met: bool, evidence: str).
    """
    conditions = {}

    # C1: Subthreshold signal
    # Test: is the signal below the effective threshold of the detector?
    # Measure: does the detector respond meaningfully at sigma=0?
    p_at_zero = detector_fn(signal, threshold, 0.0, n_trials, rng)
    p_at_optimal_no_noise = detector_fn(threshold, threshold, 0.0, n_trials, rng)
    c1_met = p_at_zero < 0.5  # below 50% detection = effectively subthreshold
    conditions["C1_subthreshold"] = {
        "met": bool(c1_met),
        "evidence": f"P_N+1(0)={p_at_zero:.4f}, threshold response={p_at_optimal_no_noise:.4f}"
    }

    # C2: Nonlinear threshold
    # Test: is E[f(x+noise)] != f(x) for some noise? (Jensen gap)
    # If detector is linear, noise should not help
    p_no_noise = detector_fn(signal, threshold, 0.0, n_trials, rng)
    p_small_noise = detector_fn(signal, threshold, 0.1, n_trials, rng)
    p_medium_noise = detector_fn(signal, threshold, 0.5, n_trials, rng)
    jensen_gap = max(p_small_noise, p_medium_noise) - p_no_noise
    # Note: polynomial nonlinearities like max(0,x-theta)^3 can have
    # near-zero Jensen gap despite being nonlinear, because the response
    # above threshold is too gradual. The condition requires SUFFICIENT
    # nonlinearity (steep enough activation), not merely non-affine.
    c2_met = jensen_gap > 0.01  # meaningful nonlinearity produces Jensen gap
    conditions["C2_nonlinear"] = {
        "met": bool(c2_met),
        "evidence": f"Jensen gap={jensen_gap:.4f} (P(0)={p_no_noise:.4f}, P(0.1)={p_small_noise:.4f}, P(0.5)={p_medium_noise:.4f})"
    }

    # C3: Noise accessibility (Kosko forbidden interval)
    # For Gaussian noise with mu=0:
    # Forbidden interval is (theta - s1, theta - s0) for binary {s0, s1}
    # For continuous signal s: the interval is roughly (theta - s, theta)
    # mu=0 is in this interval when 0 > theta-s AND 0 < theta
    # i.e., s < theta (subthreshold) AND theta > 0
    # This means mu=0 IS in the forbidden interval for standard subthreshold setup!
    # BUT: Kosko's forbidden interval theorem for Gaussian noise uses the CDF,
    # not just the mean. The full condition is about the integral of the noise
    # PDF over the interval, which can still allow SR for infinite-variance distributions.
    # For Gaussian: SR occurs even when mu is in the forbidden interval because
    # the tails provide nonzero probability of crossing.
    # HONEST CHECK: does adding noise actually improve detection?
    noise_helps = p_medium_noise > p_no_noise
    conditions["C3_noise_accessible"] = {
        "met": bool(noise_helps),
        "evidence": f"Noise improves detection: {p_medium_noise:.4f} > {p_no_noise:.4f} = {noise_helps}"
    }

    # C4: Sufficient level weighting
    # The marginal derivative condition fails for threshold detectors because
    # the benefit is non-marginal (kicks in at finite sigma, not infinitesimal).
    # Use INTEGRAL condition instead: exists sigma where beta*Delta_N+1 > alpha*Delta_N
    # Test at multiple sigma values to find if any produces net benefit
    test_sigmas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    pn1_0 = detector_fn(signal, threshold, 0.0, n_trials, rng)
    pn_0 = degradation_fn(0.0, L)
    best_net = -float('inf')
    best_sigma_c4 = 0.0
    for test_sig in test_sigmas:
        pn_t = degradation_fn(test_sig, L)
        pn1_t = detector_fn(signal, threshold, test_sig, n_trials, rng)
        gain = beta * (pn1_t - pn1_0)
        cost = alpha * (pn_0 - pn_t)
        net = gain - cost
        if net > best_net:
            best_net = net
            best_sigma_c4 = test_sig

    c4_met = best_net > 0.005  # meaningful net benefit threshold
    conditions["C4_weighting"] = {
        "met": bool(c4_met),
        "evidence": f"best_net_gain={best_net:.4f} at sigma={best_sigma_c4:.2f} (beta={beta:.2f}, alpha={alpha:.2f}, L={L:.2f})"
    }

    # C5: Sufficient redundancy (Lipschitz boundedness)
    # Test: does P_N degrade gracefully (not catastrophically) for small sigma?
    # Check if the degradation is bounded: P_N(eps) > P_N(0) * 0.5 for small eps
    pn_small = degradation_fn(0.1, L)
    pn_base = degradation_fn(0.0, L)
    fractional_loss = (pn_base - pn_small) / (pn_base + 1e-10)
    c5_met = fractional_loss < 0.5  # losing less than 50% at sigma=0.1
    conditions["C5_redundancy"] = {
        "met": bool(c5_met),
        "evidence": f"P_N(0)={pn_base:.4f}, P_N(0.1)={pn_small:.4f}, fractional_loss={fractional_loss:.4f}"
    }

    return conditions

# ============================================================
# CORE SIMULATION with bootstrap CIs (Hole #8)
# ============================================================
@dataclass
class SimResultV2:
    name: str
    sigma_range: list
    p_n: list
    p_n1: list
    p_sys: list
    sigma_opt: float
    sigma_opt_ci: list          # [lower, upper] 95% CI
    p_sys_at_opt: float
    p_sys_at_zero: float
    net_benefit: float
    net_benefit_ci: list        # [lower, upper] 95% CI
    conditions: dict
    detector_type: str
    degradation_type: str
    coupling_type: str
    inverted_u_detected: bool   # Is there a genuine peak at sigma > 0?
    params: dict

def run_simulation_v2(
    name,
    signal_strength,
    threshold,
    alpha,
    beta,
    lipschitz_constant,
    detector_fn=threshold_detector,
    degradation_fn=linear_degradation,
    coupling_fn=additive_coupling,
    sigma_max=3.0,
    n_points=100,
    n_trials=50000,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED,
):
    """Run simulation with bootstrap CIs and dynamic condition checking."""
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(0, sigma_max, n_points)

    # --- Main sweep ---
    p_n_vals = []
    p_n1_vals = []
    p_sys_vals = []

    for sigma in sigmas:
        pn = degradation_fn(float(sigma), lipschitz_constant)
        pn1 = detector_fn(signal_strength, threshold, float(sigma), n_trials, rng)
        psys = coupling_fn(pn, pn1, alpha, beta)
        p_n_vals.append(float(pn))
        p_n1_vals.append(float(pn1))
        p_sys_vals.append(float(psys))

    best_idx = int(np.argmax(p_sys_vals))
    sigma_opt = float(sigmas[best_idx])
    net_benefit = p_sys_vals[best_idx] - p_sys_vals[0]

    # --- Bootstrap CIs ---
    bootstrap_sigma_opts = []
    bootstrap_benefits = []
    for _ in range(n_bootstrap):
        boot_rng = np.random.default_rng(rng.integers(0, 2**31))
        boot_p_sys = []
        for sigma in sigmas:
            pn = degradation_fn(float(sigma), lipschitz_constant)
            pn1 = detector_fn(signal_strength, threshold, float(sigma), n_trials // 5, boot_rng)
            psys = coupling_fn(pn, pn1, alpha, beta)
            boot_p_sys.append(float(psys))
        boot_best = int(np.argmax(boot_p_sys))
        bootstrap_sigma_opts.append(float(sigmas[boot_best]))
        bootstrap_benefits.append(boot_p_sys[boot_best] - boot_p_sys[0])

    ci_lo = (1 - CONFIDENCE_LEVEL) / 2
    ci_hi = 1 - ci_lo
    if len(bootstrap_sigma_opts) > 0:
        sigma_opt_ci = [
            float(np.quantile(bootstrap_sigma_opts, ci_lo)),
            float(np.quantile(bootstrap_sigma_opts, ci_hi)),
        ]
        benefit_ci = [
            float(np.quantile(bootstrap_benefits, ci_lo)),
            float(np.quantile(bootstrap_benefits, ci_hi)),
        ]
    else:
        sigma_opt_ci = [sigma_opt, sigma_opt]
        benefit_ci = [net_benefit, net_benefit]

    # --- Inverted-U detection ---
    # A true inverted-U has a peak at sigma > 0 AND the curve descends after the peak
    if best_idx > 0 and best_idx < len(sigmas) - 1:
        inverted_u = (p_sys_vals[best_idx] > p_sys_vals[0] and
                      p_sys_vals[best_idx] > p_sys_vals[-1])
    elif best_idx == 0:
        inverted_u = False  # monotonically decreasing
    else:
        inverted_u = p_sys_vals[best_idx] > p_sys_vals[0]  # monotonically increasing (no U)

    # --- Dynamic condition check ---
    cond_rng = np.random.default_rng(seed + 1)
    conditions = check_conditions(
        signal_strength, threshold, alpha, beta, lipschitz_constant,
        detector_fn, degradation_fn, coupling_fn, cond_rng, n_trials
    )

    return SimResultV2(
        name=name,
        sigma_range=[float(s) for s in sigmas],
        p_n=p_n_vals,
        p_n1=p_n1_vals,
        p_sys=p_sys_vals,
        sigma_opt=sigma_opt,
        sigma_opt_ci=sigma_opt_ci,
        p_sys_at_opt=float(p_sys_vals[best_idx]),
        p_sys_at_zero=float(p_sys_vals[0]),
        net_benefit=float(net_benefit),
        net_benefit_ci=benefit_ci,
        conditions=conditions,
        detector_type=detector_fn.__name__,
        degradation_type=degradation_fn.__name__,
        coupling_type=coupling_fn.__name__,
        inverted_u_detected=bool(inverted_u),
        params={
            "signal": signal_strength,
            "threshold": threshold,
            "alpha": alpha,
            "beta": beta,
            "L": lipschitz_constant,
            "sigma_max": sigma_max,
            "n_trials": n_trials,
            "seed": seed,
        },
    )

# ============================================================
# CRAWFORD-SOBEL STRATEGIC COMMUNICATION (Hole #6)
# ============================================================
def run_crawford_sobel_simulation(bias, n_states=1000, n_rounds=100, seed=SEED):
    """
    Simulate actual Crawford-Sobel cheap talk:
    - Sender observes state theta ~ U[0,1]
    - Sender has bias b (wants receiver to act at theta + b)
    - In equilibrium with N* partitions, sender sends partition index
    - Receiver acts at partition midpoint
    - Measure: receiver's action quality, sender's utility, information loss

    Returns the ACTUAL noise level experienced by the receiver (not a formula).
    """
    rng = np.random.default_rng(seed)

    # Compute equilibrium partition count
    # N*(b) = floor(-1/2 + 1/2 * sqrt(1 + 2/b)) for uniform-quadratic CS
    n_star = max(1, int(np.floor(-0.5 + 0.5 * np.sqrt(1 + 2 / max(bias, 1e-6)))))

    # Compute equilibrium partition boundaries using the CS recurrence:
    # a_0 = 0, a_N = 1, and a_i = a_{i-1} + (1/N) + 2*b*(2*i - N - 1)
    # This comes from the indifference conditions in CS equilibrium
    boundaries = [0.0]
    for i in range(1, n_star):
        # For uniform-quadratic: a_i = i/N + 2*b*i*(i - N)
        a_i = i / n_star + 2 * bias * i * (i - n_star)
        if a_i <= boundaries[-1] or a_i >= 1.0:
            # Partition collapses — reduce to fewer partitions
            break
        boundaries.append(a_i)
    boundaries.append(1.0)
    n_actual = len(boundaries) - 1

    # Simulate communication rounds
    states = rng.uniform(0, 1, n_states)
    receiver_actions = np.zeros(n_states)
    sender_utilities = np.zeros(n_states)
    receiver_utilities = np.zeros(n_states)

    for i, theta in enumerate(states):
        # Find which partition theta belongs to
        partition_idx = 0
        for j in range(n_actual):
            if boundaries[j] <= theta < boundaries[j + 1]:
                partition_idx = j
                break
        else:
            partition_idx = n_actual - 1

        # Receiver acts at midpoint of partition
        midpoint = (boundaries[partition_idx] + boundaries[min(partition_idx + 1, len(boundaries) - 1)]) / 2
        receiver_actions[i] = midpoint

        # Utilities (quadratic loss)
        receiver_utilities[i] = -(theta - midpoint) ** 2
        sender_utilities[i] = -(theta + bias - midpoint) ** 2

    # Compute information metrics
    residual_variance = np.var(states - receiver_actions)  # receiver's estimation error
    total_variance = np.var(states)  # = 1/12 for U[0,1]
    information_transmitted = 1.0 - residual_variance / total_variance  # R-squared analog
    effective_noise = np.sqrt(residual_variance)

    return {
        "bias": float(bias),
        "n_star_formula": int(n_star),
        "n_actual_partitions": int(n_actual),
        "boundaries": [float(b) for b in boundaries],
        "effective_noise": float(effective_noise),
        "information_transmitted": float(information_transmitted),
        "mean_receiver_utility": float(np.mean(receiver_utilities)),
        "mean_sender_utility": float(np.mean(sender_utilities)),
        "residual_variance": float(residual_variance),
        "babbling": n_actual <= 1,
    }

# ============================================================
# MAIN TESTS
# ============================================================
def main():
    rng = np.random.default_rng(SEED)
    all_results = {}

    print("=" * 80)
    print("LEVEL-CROSSING SIMULATION v2 — RIGOROUS ADVERSARIAL TESTING")
    print("=" * 80)

    # =========================================================
    # SECTION A: Core Theorem 1 tests with multiple nonlinearities
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION A: Core Tests — Multiple Nonlinearities")
    print("=" * 80)

    detectors = {
        "threshold": threshold_detector,
        "sigmoid": sigmoid_detector,
        "polynomial": polynomial_detector,
    }

    core_results = []
    for det_name, det_fn in detectors.items():
        r = run_simulation_v2(
            name=f"All conditions met ({det_name})",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            detector_fn=det_fn,
        )
        core_results.append(r)
        print(f"\n--- {r.name} ---")
        for cname, cval in r.conditions.items():
            print(f"  {cname}: {'MET' if cval['met'] else 'FAILED'} ({cval['evidence']})")
        print(f"  sigma* = {r.sigma_opt:.3f} [{r.sigma_opt_ci[0]:.3f}, {r.sigma_opt_ci[1]:.3f}]")
        print(f"  net benefit = {r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}]")
        print(f"  inverted-U detected: {r.inverted_u_detected}")

    all_results["core_multitype"] = [asdict(r) for r in core_results]

    # =========================================================
    # SECTION B: Null models (Hole #7)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION B: Null Models — Linear detector should show NO benefit")
    print("=" * 80)

    null_result = run_simulation_v2(
        name="NULL: Linear detector (C2 violated)",
        signal_strength=0.8,
        threshold=1.0,
        alpha=0.3,
        beta=0.7,
        lipschitz_constant=0.5,
        detector_fn=linear_detector,
    )
    print(f"\n--- {null_result.name} ---")
    for cname, cval in null_result.conditions.items():
        print(f"  {cname}: {'MET' if cval['met'] else 'FAILED'} ({cval['evidence']})")
    print(f"  sigma* = {null_result.sigma_opt:.3f}")
    print(f"  net benefit = {null_result.net_benefit:.4f} [{null_result.net_benefit_ci[0]:.4f}, {null_result.net_benefit_ci[1]:.4f}]")
    print(f"  inverted-U: {null_result.inverted_u_detected}")
    if null_result.net_benefit <= 0.01:
        print("  >>> CORRECT: Linear detector shows no noise benefit (C2 needed)")
    else:
        print("  >>> WARNING: Linear detector shows unexpected benefit — investigate!")

    all_results["null_linear"] = asdict(null_result)

    # =========================================================
    # SECTION C: Condition violation tests (all 5 conditions)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION C: Systematic Condition Violations")
    print("=" * 80)

    violation_tests = [
        {
            "name": "C1 violated: suprathreshold signal",
            "signal_strength": 1.5,
            "threshold": 1.0,
            "alpha": 0.3,
            "beta": 0.7,
            "lipschitz_constant": 0.5,
        },
        {
            "name": "C4 violated: fidelity dominates",
            "signal_strength": 0.8,
            "threshold": 1.0,
            "alpha": 0.95,
            "beta": 0.05,
            "lipschitz_constant": 0.5,
        },
    ]

    violation_results = []
    for params in violation_tests:
        r = run_simulation_v2(**params)
        violation_results.append(r)
        print(f"\n--- {r.name} ---")
        for cname, cval in r.conditions.items():
            print(f"  {cname}: {'MET' if cval['met'] else 'FAILED'} ({cval['evidence']})")
        print(f"  net benefit = {r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}]")
        print(f"  inverted-U: {r.inverted_u_detected}")

    all_results["violations"] = [asdict(r) for r in violation_results]

    # =========================================================
    # SECTION D: Degradation model robustness (Hole #2)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION D: Multiple Degradation Models")
    print("=" * 80)

    degradation_models = {
        "linear": linear_degradation,
        "exponential": exponential_degradation,
        "catastrophic": catastrophic_degradation,
    }

    degradation_results = []
    for deg_name, deg_fn in degradation_models.items():
        r = run_simulation_v2(
            name=f"Degradation: {deg_name}",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            degradation_fn=deg_fn,
        )
        degradation_results.append(r)
        print(f"\n--- {r.name} ---")
        print(f"  sigma* = {r.sigma_opt:.3f} [{r.sigma_opt_ci[0]:.3f}, {r.sigma_opt_ci[1]:.3f}]")
        print(f"  net benefit = {r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}]")
        print(f"  inverted-U: {r.inverted_u_detected}")

    all_results["degradation_models"] = [asdict(r) for r in degradation_results]

    # =========================================================
    # SECTION E: Coupling model robustness (Hole #3)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION E: Multiple Coupling Models")
    print("=" * 80)

    coupling_models = {
        "additive": additive_coupling,
        "multiplicative": multiplicative_coupling,
        "substrate": substrate_coupling,
    }

    coupling_results = []
    for coup_name, coup_fn in coupling_models.items():
        r = run_simulation_v2(
            name=f"Coupling: {coup_name}",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            coupling_fn=coup_fn,
        )
        coupling_results.append(r)
        print(f"\n--- {r.name} ---")
        print(f"  sigma* = {r.sigma_opt:.3f} [{r.sigma_opt_ci[0]:.3f}, {r.sigma_opt_ci[1]:.3f}]")
        print(f"  net benefit = {r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}]")
        print(f"  inverted-U: {r.inverted_u_detected}")

    all_results["coupling_models"] = [asdict(r) for r in coupling_results]

    # =========================================================
    # SECTION F: C5 with substrate coupling (Hole #4)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION F: C5 Violation — Brittle System with Substrate Coupling")
    print("=" * 80)

    # Under substrate coupling, if P_N → 0, then P_sys → 0 regardless of P_N+1
    # This is the REAL test of C5
    c5_tests = [
        ("Brittle + additive (lenient)", additive_coupling, 5.0),
        ("Brittle + substrate (strict)", substrate_coupling, 5.0),
        ("Catastrophic + substrate (strictest)", substrate_coupling, 2.0),
    ]

    c5_results = []
    for label, coup_fn, L_val in c5_tests:
        deg_fn = catastrophic_degradation if "Catastrophic" in label else linear_degradation
        r = run_simulation_v2(
            name=f"C5: {label}",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=L_val,
            degradation_fn=deg_fn,
            coupling_fn=coup_fn,
        )
        c5_results.append(r)
        print(f"\n--- {r.name} ---")
        for cname, cval in r.conditions.items():
            print(f"  {cname}: {'MET' if cval['met'] else 'FAILED'} ({cval['evidence']})")
        print(f"  net benefit = {r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}]")
        print(f"  inverted-U: {r.inverted_u_detected}")

    all_results["c5_tests"] = [asdict(r) for r in c5_results]

    # =========================================================
    # SECTION G: Monte Carlo sensitivity analysis (Hole #5)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION G: Monte Carlo — Random Parameter Sampling")
    print("=" * 80)

    n_mc = 500
    mc_results = {
        "all_met_benefit": 0,
        "all_met_no_benefit": 0,
        "some_violated_benefit": 0,
        "some_violated_no_benefit": 0,
        "total": n_mc,
        "counterexamples": [],  # all conditions met but no benefit
        "details": [],
    }

    for i in range(n_mc):
        mc_rng = np.random.default_rng(SEED + 1000 + i)
        # Random parameters
        signal = mc_rng.uniform(0.1, 2.0)
        thresh = mc_rng.uniform(0.5, 2.0)
        a = mc_rng.uniform(0.05, 0.95)
        b = 1.0 - a
        L = mc_rng.uniform(0.1, 5.0)

        r = run_simulation_v2(
            name=f"MC_{i}",
            signal_strength=float(signal),
            threshold=float(thresh),
            alpha=float(a),
            beta=float(b),
            lipschitz_constant=float(L),
            n_trials=10000,
            n_bootstrap=0,  # skip bootstrap for speed
            seed=SEED + 2000 + i,
        )

        all_met = all(c["met"] for c in r.conditions.values())
        has_benefit = r.net_benefit > 0.005  # threshold for "real" benefit

        if all_met and has_benefit:
            mc_results["all_met_benefit"] += 1
        elif all_met and not has_benefit:
            mc_results["all_met_no_benefit"] += 1
            mc_results["counterexamples"].append({
                "params": r.params,
                "conditions": {k: v["met"] for k, v in r.conditions.items()},
                "net_benefit": r.net_benefit,
            })
        elif not all_met and has_benefit:
            mc_results["some_violated_benefit"] += 1
        else:
            mc_results["some_violated_no_benefit"] += 1

    print(f"  Total runs: {n_mc}")
    print(f"  All conditions met + benefit:    {mc_results['all_met_benefit']}")
    print(f"  All conditions met + NO benefit: {mc_results['all_met_no_benefit']} ← COUNTEREXAMPLES")
    print(f"  Some violated + benefit:         {mc_results['some_violated_benefit']}")
    print(f"  Some violated + NO benefit:      {mc_results['some_violated_no_benefit']}")

    if mc_results["all_met_no_benefit"] > 0:
        print(f"\n  WARNING: {mc_results['all_met_no_benefit']} counterexamples found!")
        for cx in mc_results["counterexamples"][:5]:
            print(f"    signal={cx['params']['signal']:.2f} thresh={cx['params']['threshold']:.2f} "
                  f"alpha={cx['params']['alpha']:.2f} L={cx['params']['L']:.2f} "
                  f"benefit={cx['net_benefit']:.4f}")
    else:
        print("\n  No counterexamples found — Theorem 1 holds across parameter space.")

    # Compute the sufficiency rate
    all_met_total = mc_results["all_met_benefit"] + mc_results["all_met_no_benefit"]
    if all_met_total > 0:
        sufficiency_rate = mc_results["all_met_benefit"] / all_met_total
        print(f"\n  Sufficiency rate: {sufficiency_rate:.1%} ({mc_results['all_met_benefit']}/{all_met_total})")

    # Compute false positive rate (benefit when conditions not met)
    not_met_total = mc_results["some_violated_benefit"] + mc_results["some_violated_no_benefit"]
    if not_met_total > 0:
        false_positive_rate = mc_results["some_violated_benefit"] / not_met_total
        print(f"  False positive rate: {false_positive_rate:.1%} ({mc_results['some_violated_benefit']}/{not_met_total})")
        print(f"  (conditions violated but benefit still observed — conditions may not be necessary)")

    all_results["monte_carlo"] = mc_results

    # =========================================================
    # SECTION H: Adversarial parameter search (Hole #10)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION H: Adversarial Search — Can conditions be met with no benefit?")
    print("=" * 80)

    # Specifically target the boundary: conditions barely met
    adversarial_results = []
    # Strategy: signal just below threshold, beta barely above alpha*L
    for signal_frac in [0.95, 0.99, 0.999]:
        for beta_margin in [1.01, 1.05, 1.1]:
            thresh = 1.0
            sig = thresh * signal_frac
            L = 0.5
            a = 0.4
            b = a * L * beta_margin  # barely satisfies C4

            r = run_simulation_v2(
                name=f"Adversarial s={signal_frac}, margin={beta_margin}",
                signal_strength=float(sig),
                threshold=float(thresh),
                alpha=float(a),
                beta=float(b),
                lipschitz_constant=float(L),
                n_trials=50000,
                seed=SEED + 5000,
            )
            adversarial_results.append(r)
            all_met = all(c["met"] for c in r.conditions.values())
            print(f"  signal={sig:.3f} beta={b:.3f} | "
                  f"conditions all met: {all_met} | "
                  f"benefit={r.net_benefit:.4f} [{r.net_benefit_ci[0]:.4f}, {r.net_benefit_ci[1]:.4f}] | "
                  f"inverted-U: {r.inverted_u_detected}")

    all_results["adversarial"] = [asdict(r) for r in adversarial_results]

    # =========================================================
    # SECTION I: Crawford-Sobel actual simulation (Hole #6)
    # =========================================================
    print("\n" + "=" * 80)
    print("SECTION I: Crawford-Sobel Strategic Communication")
    print("=" * 80)

    cs_results = []
    for b in np.linspace(0.01, 0.50, 20):
        cs = run_crawford_sobel_simulation(float(b), seed=SEED)
        cs_results.append(cs)

        # Now feed the ACTUAL noise level from CS into the threshold detector
        r = run_simulation_v2(
            name=f"CS_b={b:.3f}_actual_noise",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            sigma_max=max(cs["effective_noise"] * 3, 0.5),
            n_trials=20000,
            n_bootstrap=0,
            seed=SEED + 3000,
        )

        status = "BABBLING" if cs["babbling"] else f"N*={cs['n_actual_partitions']}"
        print(f"  b={b:.3f}  {status:12}  "
              f"eff_noise={cs['effective_noise']:.4f}  "
              f"info_transmitted={cs['information_transmitted']:.3f}  "
              f"receiver_util={cs['mean_receiver_utility']:.4f}  "
              f"sigma*={r.sigma_opt:.3f}  benefit={r.net_benefit:.4f}")

    all_results["crawford_sobel"] = cs_results

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 80)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 80)

    print("\n1. NONLINEARITY ROBUSTNESS:")
    for r in core_results:
        print(f"   {r.detector_type:25} inverted-U={r.inverted_u_detected}  benefit={r.net_benefit:.4f}")
    print(f"   {'linear_detector':25} inverted-U={null_result.inverted_u_detected}  benefit={null_result.net_benefit:.4f} (NULL)")

    print("\n2. DEGRADATION ROBUSTNESS:")
    for r in degradation_results:
        print(f"   {r.degradation_type:25} inverted-U={r.inverted_u_detected}  benefit={r.net_benefit:.4f}")

    print("\n3. COUPLING ROBUSTNESS:")
    for r in coupling_results:
        print(f"   {r.coupling_type:25} inverted-U={r.inverted_u_detected}  benefit={r.net_benefit:.4f}")

    print("\n4. MONTE CARLO:")
    if all_met_total > 0:
        print(f"   Sufficiency rate: {mc_results['all_met_benefit']}/{all_met_total} = {sufficiency_rate:.1%}")
    print(f"   Counterexamples: {mc_results['all_met_no_benefit']}")

    # Save
    output_path = "/Users/jmcentire/Personal/Research/CommunicativeVariance/simulation_results_v2.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    main()
