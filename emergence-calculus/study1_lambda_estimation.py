"""Study 1: Direct Lambda Estimation from Stigmergic Mesh Production Data.

Measures the contraction rate lambda of the Q-dynamics operator by running
the same signal stream through K fresh mesh instances with different initial
conditions, then computing Wasserstein-1 decay between empirical distributions
across runs.

Protocol (from emergence_calculus.md Section 9, Study 1):
- K >= 10 runs, same signal stream, different initial conditions
- Fit lambda from W_1 decay between pairs of runs at each time step
- Compare measured lambda with predicted lambda from Theorem 5.1

Usage:
    cd ~/WanderRepos/tools/stigmergy
    export ANTHROPIC_API_KEY=$WANDER_ANTHROPIC_API_KEY
    python ~/Personal/Research/EmergenceCalculus/study1_lambda_estimation.py

No side effects: signals are fetched read-only from live sources, mesh state
files are never written. LLM is only used for signal fetching (if needed),
not for mesh routing.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from uuid import UUID

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.optimize import curve_fit

# Must run from the stigmergy project root for config loading
STIGMERGY_ROOT = Path.home() / "WanderRepos" / "tools" / "stigmergy"
os.chdir(STIGMERGY_ROOT)
sys.path.insert(0, str(STIGMERGY_ROOT / "src"))

from stigmergy.cli.config_schema import StigmergyConfig
from stigmergy.cli.run_cmd import _load_config, _load_state, _build_sources
from stigmergy.core.familiarity import FamiliarityWeights
from stigmergy.mesh.mesh import Mesh
from stigmergy.mesh.worker import WorkerNode
from stigmergy.pipeline.processor import AgentRegistry
from stigmergy.primitives.context import Context
from stigmergy.primitives.signal import Signal
from stigmergy.services.embedding import StubEmbeddingService
from stigmergy.tracing.trace import TraceLog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

K_RUNS = 12           # Number of independent runs (>= 10 per protocol)
SNAPSHOT_EVERY = 5    # Extract state every N signals (controls granularity)

# Initial condition variations: (n_workers, term_seed_fraction)
# Each trial uses a different number of initial workers and seeds a different
# fraction of the production-learned vocabulary.
INITIAL_CONDITIONS = [
    # (n_workers, term_fraction, label)
    (1, 0.0, "1w_cold"),
    (1, 0.5, "1w_half"),
    (1, 1.0, "1w_warm"),
    (2, 0.0, "2w_cold"),
    (2, 0.5, "2w_half"),
    (2, 1.0, "2w_warm"),
    (3, 0.0, "3w_cold"),
    (3, 0.5, "3w_half"),
    (3, 1.0, "3w_warm"),
    (5, 0.0, "5w_cold"),
    (5, 0.5, "5w_half"),
    (5, 1.0, "5w_warm"),
]

assert len(INITIAL_CONDITIONS) >= K_RUNS, f"Need >= {K_RUNS} initial conditions"


# ---------------------------------------------------------------------------
# Phase 1: Fetch signals from live sources
# ---------------------------------------------------------------------------

async def fetch_signals(config: StigmergyConfig) -> list[Signal]:
    """Fetch signals from all configured live sources. Read-only."""
    sources = _build_sources(config, live=True)
    state = _load_state()

    # Use a 14-day window to get a meaningful signal volume
    since = datetime.now(timezone.utc) - timedelta(days=14)
    last_run = state.get("last_run")
    if last_run:
        lr = datetime.fromisoformat(last_run)
        # Use whichever is older to get more signals
        since = min(since, lr)

    all_signals: list[Signal] = []
    for source_name, adapter, is_live in sources:
        print(f"  Fetching {source_name} ({'live' if is_live else 'mock'})...")
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            print(f"    SKIP: {source_name} connection failed: {e}")
            continue
        count = 0
        async for sig in adapter.backfill(since):
            all_signals.append(sig)
            count += 1
        print(f"    {source_name}: {count} signals")

    # Sort chronologically (oldest first) — same as production pipeline
    all_signals.sort(key=lambda s: s.timestamp)
    print(f"\n  Total: {len(all_signals)} signals")
    return all_signals


# ---------------------------------------------------------------------------
# Phase 2: Build fresh mesh and replay signals
# ---------------------------------------------------------------------------

def build_mesh(
    config: StigmergyConfig,
    n_workers: int,
    term_fraction: float,
) -> Mesh:
    """Build a fresh mesh with specified initial conditions.

    n_workers: number of initial workers
    term_fraction: fraction of production-learned terms to seed (0.0 = cold, 1.0 = warm)
    """
    state = _load_state()
    mc = config.mesh
    wc = mc.worker
    fw = config.pipeline.familiarity_weights

    weights = FamiliarityWeights(
        embedding_similarity=fw.embedding_similarity,
        keyword_overlap=fw.keyword_overlap,
        source_affinity=fw.source_affinity,
        temporal_proximity=fw.temporal_proximity,
        author_affinity=fw.author_affinity,
    )

    mesh = Mesh(
        AgentRegistry(),
        embedding_service=StubEmbeddingService(),
        trace_log=TraceLog(),
        max_hops=mc.max_hops,
        max_workers=mc.max_workers,
        familiarity_weights=weights,
        consensus_threshold=config.pipeline.consensus_threshold,
        uncertainty_threshold=config.pipeline.uncertainty_threshold,
        dedup_enabled=False,  # No dedup for experiment — same signals, different runs
        worker_capacity=wc.capacity,
        base_threshold=wc.base_threshold,
        max_threshold=wc.max_threshold,
        threshold_curve=wc.threshold_curve,
        high_relevance_offset=wc.high_relevance_offset,
        # No LLM — pure mechanical routing for reproducibility
    )

    # Create workers. If term_fraction > 0, seed from production vocabulary.
    context_configs = list(config.contexts.items())
    saved_state = state.get("context_state", {})

    for i in range(n_workers):
        # Cycle through configured contexts for seeding
        if context_configs:
            ctx_name, ctx_cfg = context_configs[i % len(context_configs)]
            saved = saved_state.get(ctx_name, {})
            all_terms = set(ctx_cfg.terms)
            all_terms.update(saved.get("learned_terms", []))

            # Sample terms based on fraction
            if term_fraction > 0 and all_terms:
                rng = np.random.RandomState(seed=i * 1000 + int(term_fraction * 100))
                term_list = sorted(all_terms)  # deterministic ordering
                n_terms = max(1, int(len(term_list) * term_fraction))
                selected = set(rng.choice(term_list, size=n_terms, replace=False))
            else:
                selected = set()

            ctx = Context(
                capacity=wc.capacity,
                business_weight=ctx_cfg.business_weight,
            )
            ctx.terms = selected
            ctx.term_bloom.add_many(selected)

            if term_fraction > 0 and saved:
                # Also seed source/author distributions proportionally
                for src, cnt in saved.get("source_counts", {}).items():
                    ctx.source_counts[src] = max(1, int(cnt * term_fraction))
                for auth, cnt in saved.get("author_counts", {}).items():
                    ctx.author_counts[auth] = max(1, int(cnt * term_fraction))

            ctx.last_signal = datetime.now(timezone.utc)
        else:
            ctx = Context(capacity=wc.capacity)

        worker = WorkerNode(
            ctx,
            base_threshold=wc.base_threshold,
            max_threshold=wc.max_threshold,
            threshold_curve=wc.threshold_curve,
            high_relevance_offset=wc.high_relevance_offset,
        )
        mesh.add_worker(worker)

    # Connect workers (fully connected for small counts, ring for larger)
    worker_ids = [w.id for w in mesh.workers]
    if len(worker_ids) <= 5:
        for i, a in enumerate(worker_ids):
            for b in worker_ids[i + 1:]:
                mesh.connect(a, b)
    else:
        for i in range(len(worker_ids)):
            mesh.connect(worker_ids[i], worker_ids[(i + 1) % len(worker_ids)])

    return mesh


def extract_state(mesh: Mesh) -> dict:
    """Extract the empirical distribution state from a mesh.

    Returns a feature vector per worker that characterizes Q_t:
    - term count, signal count, vigilance, fullness, avg familiarity
    - source distribution entropy
    - worker count
    """
    workers = mesh.workers
    state = {
        "worker_count": len(workers),
        "total_signals": mesh.signals_ingested,
        "workers": [],
    }
    for w in workers:
        src_counts = dict(w.context.source_counts)
        total = sum(src_counts.values()) if src_counts else 0
        if total > 0:
            probs = np.array([c / total for c in src_counts.values()])
            src_entropy = -np.sum(probs * np.log2(probs + 1e-12))
        else:
            src_entropy = 0.0

        state["workers"].append({
            "signal_count": w.context.signal_count,
            "fullness": w.fullness,
            "vigilance": w.adaptive_threshold,
            "avg_familiarity": w.rolling_avg_familiarity,
            "num_terms": len(w.context.terms),
            "energy": w.context.energy,
            "source_entropy": src_entropy,
            "accepted": w.signals_accepted,
            "forwarded": w.signals_forwarded,
        })
    return state


def state_to_distribution(state: dict) -> np.ndarray:
    """Convert mesh state snapshot to a 1D empirical distribution for W_1.

    Uses the per-worker feature vectors as weighted points in feature space.
    The distribution is the set of (signal_count-weighted) vigilance +
    avg_familiarity + source_entropy values per worker.

    For W_1, we need a 1D distribution. We use a composite score:
        score = 0.4 * vigilance + 0.3 * avg_familiarity + 0.3 * normalized_terms

    Each worker contributes `signal_count` copies of its score (weighting by
    how much data it has absorbed).
    """
    points = []
    for w in state["workers"]:
        # Composite score capturing Q's location
        score = (
            0.4 * w["vigilance"]
            + 0.3 * w["avg_familiarity"]
            + 0.3 * min(1.0, w["num_terms"] / 100.0)  # normalize term count
        )
        # Weight by signals absorbed (minimum 1 for workers that exist but haven't accepted)
        weight = max(1, w["signal_count"])
        points.extend([score] * weight)
    return np.array(points) if points else np.array([0.0])


async def run_trial(
    signals: list[Signal],
    config: StigmergyConfig,
    n_workers: int,
    term_fraction: float,
    label: str,
) -> list[dict]:
    """Run one trial: feed all signals through a fresh mesh, snapshot periodically."""
    mesh = build_mesh(config, n_workers, term_fraction)
    snapshots = []

    print(f"    {label}: {mesh.worker_count} workers, {len(signals)} signals...", end="", flush=True)
    t0 = time.monotonic()

    for i, signal in enumerate(signals):
        await mesh.ingest(signal)

        if (i + 1) % SNAPSHOT_EVERY == 0 or i == len(signals) - 1:
            state = extract_state(mesh)
            state["signal_idx"] = i
            state["label"] = label
            snapshots.append(state)

    elapsed = time.monotonic() - t0
    print(f" done ({elapsed:.1f}s, {mesh.worker_count} workers final)")
    return snapshots


# ---------------------------------------------------------------------------
# Phase 3: Compute W_1 decay and fit lambda
# ---------------------------------------------------------------------------

def compute_w1_matrix(all_snapshots: dict[str, list[dict]]) -> dict:
    """Compute W_1 between all pairs of runs at each snapshot index.

    Returns {snapshot_idx: {(label_i, label_j): w1_value}}.
    """
    labels = list(all_snapshots.keys())
    # All runs have the same number of snapshots
    n_snapshots = min(len(snaps) for snaps in all_snapshots.values())

    w1_results = {}
    for t in range(n_snapshots):
        w1_results[t] = {}
        for li, lj in combinations(labels, 2):
            dist_i = state_to_distribution(all_snapshots[li][t])
            dist_j = state_to_distribution(all_snapshots[lj][t])
            w1 = wasserstein_distance(dist_i, dist_j)
            w1_results[t][(li, lj)] = w1

    return w1_results


def fit_lambda(w1_matrix: dict) -> dict:
    """Fit exponential decay lambda from W_1 time series.

    Model: W_1(t) = W_1(0) * lambda^t

    Returns fit parameters and diagnostics.
    """
    timesteps = sorted(w1_matrix.keys())
    if len(timesteps) < 3:
        return {"error": "Too few timesteps for fitting"}

    # Average W_1 across all pairs at each timestep
    mean_w1 = []
    for t in timesteps:
        values = list(w1_matrix[t].values())
        mean_w1.append(np.mean(values) if values else 0.0)

    mean_w1 = np.array(mean_w1)
    t_values = np.array(timesteps, dtype=float)

    # Normalize time to signal indices
    t_norm = t_values / t_values[-1] if t_values[-1] > 0 else t_values

    # Filter out zero values for log fitting
    nonzero = mean_w1 > 1e-10
    if np.sum(nonzero) < 3:
        return {
            "error": "W_1 converged too quickly (< 3 nonzero points)",
            "mean_w1": mean_w1.tolist(),
            "timesteps": timesteps,
        }

    # Fit exponential: W_1(t) = A * exp(-gamma * t)
    # In log space: log(W_1) = log(A) - gamma * t
    log_w1 = np.log(mean_w1[nonzero])
    t_fit = t_norm[nonzero]

    try:
        # Linear fit in log space
        coeffs = np.polyfit(t_fit, log_w1, 1)
        gamma = -coeffs[0]
        A = np.exp(coeffs[1])
        lambda_estimated = np.exp(-gamma) if gamma > 0 else 1.0

        # Also try direct nonlinear fit for robustness
        def exp_model(t, a, lam):
            return a * np.power(lam, t)

        try:
            popt, pcov = curve_fit(
                exp_model, t_fit, mean_w1[nonzero],
                p0=[A, lambda_estimated],
                bounds=([0, 0], [np.inf, 1.0]),
                maxfev=10000,
            )
            lambda_nls = popt[1]
            lambda_se = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else float("nan")
        except (RuntimeError, ValueError):
            lambda_nls = float("nan")
            lambda_se = float("nan")

        # Compute R^2
        predicted = A * np.power(lambda_estimated, t_fit)
        ss_res = np.sum((mean_w1[nonzero] - predicted) ** 2)
        ss_tot = np.sum((mean_w1[nonzero] - np.mean(mean_w1[nonzero])) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            "lambda_log_fit": float(lambda_estimated),
            "lambda_nls_fit": float(lambda_nls),
            "lambda_nls_se": float(lambda_se),
            "gamma": float(gamma),
            "A": float(A),
            "r_squared": float(r_squared),
            "n_nonzero_points": int(np.sum(nonzero)),
            "n_total_points": len(timesteps),
            "mean_w1": mean_w1.tolist(),
            "timesteps": timesteps,
            "w1_initial": float(mean_w1[0]),
            "w1_final": float(mean_w1[-1]),
            "convergence_ratio": float(mean_w1[-1] / mean_w1[0]) if mean_w1[0] > 0 else 0.0,
        }

    except (np.linalg.LinAlgError, ValueError) as e:
        return {
            "error": str(e),
            "mean_w1": mean_w1.tolist(),
            "timesteps": timesteps,
        }


def compute_predicted_lambda(config: StigmergyConfig) -> dict:
    """Compute predicted lambda from Theorem 5.1 using mesh parameters.

    The Gaussian prediction: lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa)
    where sigma^2_* is the variance fixed point.

    For the mesh, sigma^2_kappa relates to scoring function curvature and
    sigma^2_L to lifecycle variance injection. These are approximated from
    the mesh's vigilance parameters.
    """
    wc = config.mesh.worker

    # Approximate sigma^2_kappa from the vigilance threshold range.
    # The scoring function's effective width maps to the gap between
    # base_threshold and max_threshold. A wider gap = broader scoring = larger sigma^2_kappa.
    sigma_kappa_sq = (wc.max_threshold - wc.base_threshold) ** 2

    # Approximate sigma^2_L from lifecycle variance injection.
    # In the mesh, variance injection comes from:
    # 1. New signal diversity (approximated by gap_threshold)
    # 2. Worker spawning/decay cycles
    # Conservative estimate: gap_threshold as proxy for noise floor
    sigma_L_sq = wc.base_threshold ** 2  # Lower bound on lifecycle noise

    if sigma_kappa_sq <= 0 or sigma_L_sq <= 0:
        return {"error": "degenerate parameters"}

    # Fixed point variance: sigma^2_* = (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa)) / 2
    discriminant = sigma_L_sq ** 2 + 4 * sigma_L_sq * sigma_kappa_sq
    sigma_star_sq = (sigma_L_sq + np.sqrt(discriminant)) / 2

    # Theorem 5.1 contraction parameter
    lambda_star = sigma_kappa_sq / (sigma_star_sq + sigma_kappa_sq)

    # Uniform (worst-case) rate
    lambda_uniform = sigma_kappa_sq / (sigma_L_sq + sigma_kappa_sq)

    return {
        "sigma_kappa_sq": float(sigma_kappa_sq),
        "sigma_L_sq": float(sigma_L_sq),
        "sigma_star_sq": float(sigma_star_sq),
        "lambda_star": float(lambda_star),
        "lambda_uniform": float(lambda_uniform),
        "base_threshold": wc.base_threshold,
        "max_threshold": wc.max_threshold,
    }


# ---------------------------------------------------------------------------
# Per-pair W_1 decay analysis
# ---------------------------------------------------------------------------

def analyze_pair_convergence(all_snapshots: dict[str, list[dict]]) -> list[dict]:
    """Analyze convergence patterns for specific pairs of interest.

    Groups: same-workers-different-warmth, same-warmth-different-workers.
    """
    analyses = []
    labels = list(all_snapshots.keys())
    n_snapshots = min(len(snaps) for snaps in all_snapshots.values())

    for li, lj in combinations(labels, 2):
        w1_series = []
        for t in range(n_snapshots):
            di = state_to_distribution(all_snapshots[li][t])
            dj = state_to_distribution(all_snapshots[lj][t])
            w1_series.append(float(wasserstein_distance(di, dj)))

        # Classify pair
        ni, ti = li.split("_")[0], li.split("_")[1]
        nj, tj = lj.split("_")[0], lj.split("_")[1]

        if ni == nj:
            pair_type = "same_workers"
        elif ti == tj:
            pair_type = "same_warmth"
        else:
            pair_type = "mixed"

        analyses.append({
            "pair": (li, lj),
            "pair_type": pair_type,
            "w1_initial": w1_series[0] if w1_series else 0,
            "w1_final": w1_series[-1] if w1_series else 0,
            "convergence_ratio": w1_series[-1] / w1_series[0] if w1_series and w1_series[0] > 0 else 0,
            "w1_series": w1_series,
        })

    return analyses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70)
    print("Study 1: Direct Lambda Estimation")
    print("Emergence Design Calculus — Empirical Validation")
    print("=" * 70)
    print()

    # Load production config
    print("[1/4] Loading production config...")
    config = _load_config()
    print(f"  Config loaded: {len(config.contexts)} contexts, "
          f"{len(config.agents)} agents, "
          f"{config.mesh.max_workers} max workers")
    print()

    # Compute predicted lambda
    print("[2/4] Computing predicted lambda from Theorem 5.1...")
    predicted = compute_predicted_lambda(config)
    if "error" not in predicted:
        print(f"  sigma^2_kappa = {predicted['sigma_kappa_sq']:.6f}")
        print(f"  sigma^2_L     = {predicted['sigma_L_sq']:.6f}")
        print(f"  sigma^2_*     = {predicted['sigma_star_sq']:.6f}")
        print(f"  lambda_*      = {predicted['lambda_star']:.6f} (asymptotic)")
        print(f"  lambda_uniform= {predicted['lambda_uniform']:.6f} (worst-case)")
    else:
        print(f"  Error: {predicted['error']}")
    print()

    # Fetch signals
    print("[3/4] Fetching production signals...")
    signals = await fetch_signals(config)
    if not signals:
        print("  ERROR: No signals fetched. Check API keys and connectivity.")
        return
    print()

    # Run K trials
    print(f"[4/4] Running {K_RUNS} trials...")
    all_snapshots: dict[str, list[dict]] = {}

    for ic in INITIAL_CONDITIONS[:K_RUNS]:
        n_workers, term_frac, label = ic
        snapshots = await run_trial(signals, config, n_workers, term_frac, label)
        all_snapshots[label] = snapshots

    print()

    # Compute W_1 matrix
    print("Computing W_1 decay across all pairs...")
    w1_matrix = compute_w1_matrix(all_snapshots)

    # Fit lambda
    print("Fitting exponential decay...")
    fit_result = fit_lambda(w1_matrix)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if "error" not in fit_result:
        print(f"  lambda (log-linear fit):   {fit_result['lambda_log_fit']:.6f}")
        print(f"  lambda (nonlinear fit):    {fit_result['lambda_nls_fit']:.6f} "
              f"+/- {fit_result['lambda_nls_se']:.6f}")
        print(f"  R^2:                       {fit_result['r_squared']:.4f}")
        print(f"  W_1 initial:               {fit_result['w1_initial']:.6f}")
        print(f"  W_1 final:                 {fit_result['w1_final']:.6f}")
        print(f"  Convergence ratio:         {fit_result['convergence_ratio']:.6f}")
        print(f"  Nonzero data points:       {fit_result['n_nonzero_points']}/{fit_result['n_total_points']}")

        if "error" not in predicted:
            print()
            print("  COMPARISON WITH THEOREM 5.1:")
            print(f"    Predicted lambda_*:      {predicted['lambda_star']:.6f}")
            print(f"    Predicted lambda_uniform: {predicted['lambda_uniform']:.6f}")
            print(f"    Measured lambda:          {fit_result['lambda_nls_fit']:.6f}")

            # Diagnostic
            measured = fit_result['lambda_nls_fit']
            if not np.isnan(measured):
                if measured < predicted['lambda_star']:
                    print("    Status: Measured < predicted (contraction FASTER than Gaussian prediction)")
                elif measured < predicted['lambda_uniform']:
                    print("    Status: lambda_* < measured < lambda_uniform (consistent with theory)")
                elif measured < 1.0:
                    print("    Status: Contraction observed but slower than Gaussian prediction")
                else:
                    print("    Status: NO contraction observed (lambda >= 1)")
    else:
        print(f"  Fitting error: {fit_result['error']}")
        if "mean_w1" in fit_result:
            print(f"  W_1 series: {fit_result['mean_w1'][:10]}...")

    # Pair analysis
    print()
    print("Pair convergence analysis:")
    pair_analyses = analyze_pair_convergence(all_snapshots)

    for ptype in ["same_workers", "same_warmth", "mixed"]:
        pairs = [p for p in pair_analyses if p["pair_type"] == ptype]
        if pairs:
            ratios = [p["convergence_ratio"] for p in pairs]
            print(f"  {ptype}: n={len(pairs)}, "
                  f"mean convergence ratio={np.mean(ratios):.4f}, "
                  f"min={np.min(ratios):.4f}, max={np.max(ratios):.4f}")

    # Save full results
    output_path = Path.home() / "Personal" / "Research" / "EmergenceCalculus" / "study1_results.json"
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "k_runs": K_RUNS,
            "snapshot_every": SNAPSHOT_EVERY,
            "n_signals": len(signals),
            "initial_conditions": [
                {"n_workers": ic[0], "term_fraction": ic[1], "label": ic[2]}
                for ic in INITIAL_CONDITIONS[:K_RUNS]
            ],
        },
        "predicted_lambda": predicted,
        "fit_result": {
            k: v for k, v in fit_result.items()
            if k != "mean_w1"  # keep JSON small
        },
        "mean_w1_series": fit_result.get("mean_w1", []),
        "pair_analyses": [
            {
                "pair": p["pair"],
                "pair_type": p["pair_type"],
                "w1_initial": p["w1_initial"],
                "w1_final": p["w1_final"],
                "convergence_ratio": p["convergence_ratio"],
                # Include full series for detailed plotting
                "w1_series": p["w1_series"],
            }
            for p in pair_analyses
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
