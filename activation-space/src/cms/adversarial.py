"""Adversarial analysis for the coverage-clustering tradeoff.

Formalizes the fundamental tension: to cover the capability manifold
efficiently, the attacker must sample diverse regions; to avoid detection,
they must cluster their queries. These goals are in direct conflict.

This module computes:
1. Coverage efficiency under detection constraints
2. The C(tau_mu) bound: maximum coverage as a function of similarity threshold
3. Pareto frontier of coverage vs. evasion
"""

import numpy as np
from dataclasses import dataclass
from .sessions import Session, compute_coverage
from .detectors import _pairwise_centered_cosine, _similarity_entropy


@dataclass
class AdversarialResult:
    """Result of adversarial efficiency analysis."""
    tau_mu: float                # Detection threshold
    coverage: float              # Manifold coverage achieved
    mean_similarity: float       # Actual mean pairwise similarity
    entropy: float               # Similarity distribution entropy
    n_queries: int               # Number of queries used
    efficiency: float            # coverage / n_queries (normalized)
    detected: bool               # Whether detection threshold is crossed


def unconstrained_coverage(
    fingerprints: np.ndarray,
    k: int = 50,
    n_trials: int = 100,
    rng: np.random.RandomState = None,
) -> float:
    """Compute baseline unconstrained coverage (farthest-first, no evasion).

    This is the maximum achievable coverage with k queries when the
    attacker doesn't care about detection.
    """
    if rng is None:
        rng = np.random.RandomState(200)

    n_probes = fingerprints.shape[0]
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8
    fp_normed = fingerprints / norms

    coverages = []
    for _ in range(n_trials):
        start = rng.randint(0, n_probes)
        selected = [start]
        remaining = set(range(n_probes)) - {start}

        for _ in range(k - 1):
            if not remaining:
                break
            rem_list = list(remaining)
            sel_fps = fp_normed[selected]
            rem_fps = fp_normed[rem_list]
            sims = rem_fps @ sel_fps.T
            min_sims = sims.max(axis=1)
            best_idx = rem_list[np.argmin(min_sims)]
            selected.append(best_idx)
            remaining.discard(best_idx)

        session = Session(probe_indices=selected[:k], session_type='baseline', label=1)
        cov = compute_coverage(session, fingerprints)
        coverages.append(cov)

    return float(np.mean(coverages))


def constrained_coverage(
    fingerprints: np.ndarray,
    tau_mu: float,
    k: int = 50,
    n_trials: int = 100,
    rng: np.random.RandomState = None,
) -> AdversarialResult:
    """Compute coverage achievable under similarity constraint.

    The attacker must maintain mean pairwise centered cosine similarity
    above tau_mu. This forces clustering, reducing coverage.

    Uses greedy strategy: maximize coverage while checking constraint
    after each addition.
    """
    if rng is None:
        rng = np.random.RandomState(201)

    n_probes = fingerprints.shape[0]
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8
    fp_normed = fingerprints / norms

    # Center for Pearson correlation
    centered = fingerprints - fingerprints.mean(axis=1, keepdims=True)
    c_norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
    fp_centered = centered / c_norms

    # Precompute full centered similarity matrix for fast constraint checking
    sim_matrix = fp_centered @ fp_centered.T

    results = []

    for _ in range(n_trials):
        start = rng.randint(0, n_probes)
        selected = [start]
        remaining = set(range(n_probes)) - {start}

        # Track running sum of pairwise similarities for O(k) constraint check
        pairwise_sum = 0.0
        n_pairs = 0

        for _ in range(k - 1):
            if not remaining:
                break

            # Farthest-first: rank candidates by distance from selected
            rem_list = list(remaining)
            sel_fps = fp_normed[selected]
            rem_fps = fp_normed[rem_list]
            sims = rem_fps @ sel_fps.T
            max_sims = sims.max(axis=1)  # closest selected for each remaining
            # Sort by ascending max-similarity (farthest first)
            order = np.argsort(max_sims)

            added = False
            for idx in order:
                candidate = rem_list[idx]
                # Check constraint: would adding this query keep mu >= tau_mu?
                # Incremental check: new_sum = old_sum + sum of sims to all selected
                new_sims = sim_matrix[candidate, selected]
                new_pair_sum = pairwise_sum + float(new_sims.sum())
                new_n_pairs = n_pairs + len(selected)
                new_mean = new_pair_sum / new_n_pairs if new_n_pairs > 0 else 1.0

                if new_mean >= tau_mu:
                    selected.append(candidate)
                    remaining.discard(candidate)
                    pairwise_sum = new_pair_sum
                    n_pairs = new_n_pairs
                    added = True
                    break

            if not added:
                # No candidate satisfies constraint — stop
                break

        session = Session(probe_indices=selected, session_type='constrained', label=1)
        cov = compute_coverage(session, fingerprints)

        # Compute final statistics
        final_fps = fp_centered[selected]
        final_sims = _pairwise_centered_cosine(fingerprints[selected])

        results.append(AdversarialResult(
            tau_mu=tau_mu,
            coverage=cov,
            mean_similarity=float(final_sims.mean()) if len(final_sims) > 0 else 0.0,
            entropy=_similarity_entropy(final_sims),
            n_queries=len(selected),
            efficiency=cov / max(len(selected), 1),
            detected=float(final_sims.mean()) < tau_mu if len(final_sims) > 0 else True,
        ))

    # Average across trials
    avg_coverage = float(np.mean([r.coverage for r in results]))
    avg_mu = float(np.mean([r.mean_similarity for r in results]))
    avg_entropy = float(np.mean([r.entropy for r in results]))
    avg_n = float(np.mean([r.n_queries for r in results]))
    avg_eff = float(np.mean([r.efficiency for r in results]))
    any_detected = any(r.detected for r in results)

    return AdversarialResult(
        tau_mu=tau_mu,
        coverage=avg_coverage,
        mean_similarity=avg_mu,
        entropy=avg_entropy,
        n_queries=int(avg_n),
        efficiency=avg_eff,
        detected=any_detected,
    )


def coverage_efficiency_sweep(
    fingerprints: np.ndarray,
    tau_values: list[float] = None,
    k: int = 50,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """Sweep tau_mu values and compute coverage-efficiency curve.

    This produces the key Figure 3: how extraction efficiency drops
    as detection threshold tightens.

    Returns:
        dict with 'tau_values', 'coverages', 'efficiencies',
        'baseline_coverage', 'efficiency_ratios'
    """
    if tau_values is None:
        tau_values = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70,
                      0.75, 0.80, 0.85, 0.90, 0.95]

    rng = np.random.RandomState(seed)

    # Baseline: unconstrained coverage
    baseline = unconstrained_coverage(
        fingerprints, k=k, n_trials=n_trials,
        rng=np.random.RandomState(rng.randint(0, 2**31)))

    results = []
    for tau in tau_values:
        if tau <= 0.0:
            # No constraint = baseline
            result = AdversarialResult(
                tau_mu=0.0, coverage=baseline, mean_similarity=0.0,
                entropy=0.0, n_queries=k, efficiency=baseline/k,
                detected=False)
        else:
            result = constrained_coverage(
                fingerprints, tau_mu=tau, k=k, n_trials=n_trials,
                rng=np.random.RandomState(rng.randint(0, 2**31)))
        results.append(result)

    coverages = [r.coverage for r in results]
    efficiencies = [r.efficiency for r in results]
    efficiency_ratios = [c / baseline if baseline > 0 else 0.0 for c in coverages]

    return {
        'tau_values': tau_values,
        'coverages': coverages,
        'efficiencies': efficiencies,
        'baseline_coverage': baseline,
        'efficiency_ratios': efficiency_ratios,
        'results': results,
    }


def attacker_cost_model(
    sweep_results: dict,
    target_coverage: float = 0.8,
) -> dict:
    """Estimate attacker cost under detection constraints.

    Given a target coverage fraction, compute how many queries are needed
    at each constraint level compared to unconstrained.

    Returns:
        dict with cost multipliers at each tau level
    """
    baseline = sweep_results['baseline_coverage']
    target_absolute = target_coverage * baseline

    costs = {}
    for tau, cov, eff in zip(
        sweep_results['tau_values'],
        sweep_results['coverages'],
        sweep_results['efficiencies'],
    ):
        if cov > 0:
            # How many sessions needed to reach target coverage
            # (assuming coverage accumulates linearly — optimistic for attacker)
            sessions_needed = target_absolute / cov if cov > 0 else float('inf')
            cost_multiplier = sessions_needed  # relative to 1 unconstrained session
        else:
            cost_multiplier = float('inf')
        costs[tau] = {
            'sessions_needed': sessions_needed,
            'cost_multiplier': cost_multiplier,
            'achievable_coverage': cov,
        }

    return costs
