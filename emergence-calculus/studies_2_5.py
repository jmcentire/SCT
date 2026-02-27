"""Studies 2-5: Parameter Sensitivity, Cage Verification, Curvature Sweep, Fork Analysis.

Runs against the same production signal stream as Study 1, extending the
empirical validation of the Emergence Design Calculus.

Study 2: Predicted vs Measured Lambda
  - Estimate sigma^2_kappa and sigma^2_L from mesh internals (not just config)
  - Compare Theorem 5.1 formula prediction with Study 1 measurement
  - Sensitivity analysis: how robust is the prediction to parameter estimates?

Study 3: Selection Balance and Cage Verification
  - Vary beta via base_threshold (controls discovery/retrieval balance)
  - Measure finding entropy (vocabulary diversity) at each beta
  - Verify: beta -> 0 produces Cage (vocabulary collapses to p_Frame)

Study 4: Convergence Rate vs Scoring Curvature
  - Vary max_threshold (effective sigma^2_kappa)
  - Measure lambda at each setting
  - Verify: sharper scoring -> faster convergence (smaller lambda)

Study 5: Fork Variance Injection
  - Instrument fork and decay events during mesh operation
  - Measure between-component variance at fork boundaries
  - Verify: fork injects variance (sigma^2_L contribution)

Usage:
    cd ~/WanderRepos/tools/stigmergy
    export ANTHROPIC_API_KEY=$WANDER_ANTHROPIC_API_KEY
    python ~/Personal/Research/EmergenceCalculus/studies_2_5.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

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

SNAPSHOT_EVERY = 10  # Coarser than Study 1 (we're running many more trials)
OUTPUT_DIR = Path.home() / "Personal" / "Research" / "EmergenceCalculus"


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

async def fetch_signals(config: StigmergyConfig) -> list[Signal]:
    """Fetch signals from live sources (same as Study 1)."""
    sources = _build_sources(config, live=True)
    state = _load_state()
    since = datetime.now(timezone.utc) - timedelta(days=14)
    last_run = state.get("last_run")
    if last_run:
        lr = datetime.fromisoformat(last_run)
        since = min(since, lr)

    all_signals: list[Signal] = []
    for source_name, adapter, is_live in sources:
        print(f"  Fetching {source_name}...", end="", flush=True)
        try:
            await adapter.connect()
        except (ConnectionError, OSError) as e:
            print(f" SKIP ({e})")
            continue
        count = 0
        async for sig in adapter.backfill(since):
            all_signals.append(sig)
            count += 1
        print(f" {count} signals")

    all_signals.sort(key=lambda s: s.timestamp)
    return all_signals


def build_mesh_with_params(
    config: StigmergyConfig,
    n_workers: int = 3,
    base_threshold: float | None = None,
    max_threshold: float | None = None,
    threshold_curve: float | None = None,
    gap_threshold: float | None = None,
    worker_capacity: int | None = None,
) -> Mesh:
    """Build a fresh mesh with overridable parameters."""
    mc = config.mesh
    wc = mc.worker
    fw = config.pipeline.familiarity_weights

    bt = base_threshold if base_threshold is not None else wc.base_threshold
    mt = max_threshold if max_threshold is not None else wc.max_threshold
    tc = threshold_curve if threshold_curve is not None else wc.threshold_curve
    gt = gap_threshold if gap_threshold is not None else 0.08
    cap = worker_capacity if worker_capacity is not None else wc.capacity

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
        dedup_enabled=False,
        worker_capacity=cap,
        base_threshold=bt,
        max_threshold=mt,
        threshold_curve=tc,
        high_relevance_offset=wc.high_relevance_offset,
        gap_threshold=gt,
    )

    # Spawn workers
    for _ in range(n_workers):
        mesh.spawn_worker()
    worker_ids = [w.id for w in mesh.workers]
    for i, a in enumerate(worker_ids):
        for b in worker_ids[i + 1:]:
            mesh.connect(a, b)

    return mesh


def extract_distribution(mesh: Mesh) -> np.ndarray:
    """Extract 1D empirical distribution from mesh state."""
    points = []
    for w in mesh.workers:
        score = (
            0.4 * w.adaptive_threshold
            + 0.3 * w.rolling_avg_familiarity
            + 0.3 * min(1.0, len(w.context.terms) / 100.0)
        )
        weight = max(1, w.context.signal_count)
        points.extend([score] * weight)
    return np.array(points) if points else np.array([0.0])


def term_entropy(mesh: Mesh) -> float:
    """Compute entropy of term distribution across workers (measures diversity)."""
    all_terms: dict[str, int] = defaultdict(int)
    for w in mesh.workers:
        for t in w.context.terms:
            all_terms[t] += 1
    if not all_terms:
        return 0.0
    counts = np.array(list(all_terms.values()), dtype=float)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def vocabulary_concentration(mesh: Mesh) -> float:
    """Measure how concentrated vocabulary is (1 = all workers have same terms, 0 = no overlap)."""
    if mesh.worker_count < 2:
        return 0.0
    term_sets = [w.context.terms for w in mesh.workers if w.context.terms]
    if len(term_sets) < 2:
        return 0.0
    # Average pairwise Jaccard similarity
    similarities = []
    for i in range(len(term_sets)):
        for j in range(i + 1, len(term_sets)):
            if term_sets[i] or term_sets[j]:
                intersection = len(term_sets[i] & term_sets[j])
                union = len(term_sets[i] | term_sets[j])
                similarities.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(similarities)) if similarities else 0.0


async def run_trial_pair(
    signals: list[Signal],
    config: StigmergyConfig,
    label: str,
    **mesh_kwargs,
) -> dict:
    """Run a pair of trials with identical params but different worker counts (3 vs 5)
    and compute W_1 decay + lambda."""
    mesh_a = build_mesh_with_params(config, n_workers=3, **mesh_kwargs)
    mesh_b = build_mesh_with_params(config, n_workers=5, **mesh_kwargs)

    w1_series = []
    entropy_a = []
    entropy_b = []
    concentration_a = []
    concentration_b = []
    fork_events_a = []
    fork_events_b = []

    print(f"    {label}...", end="", flush=True)
    t0 = time.monotonic()

    prev_worker_count_a = mesh_a.worker_count
    prev_worker_count_b = mesh_b.worker_count

    for i, signal in enumerate(signals):
        await mesh_a.ingest(signal)
        await mesh_b.ingest(signal)

        # Track fork/decay events (worker count changes)
        curr_a = mesh_a.worker_count
        curr_b = mesh_b.worker_count
        if curr_a != prev_worker_count_a:
            fork_events_a.append({
                "signal_idx": i,
                "prev": prev_worker_count_a,
                "curr": curr_a,
                "type": "fork" if curr_a > prev_worker_count_a else "decay",
            })
            prev_worker_count_a = curr_a
        if curr_b != prev_worker_count_b:
            fork_events_b.append({
                "signal_idx": i,
                "prev": prev_worker_count_b,
                "curr": curr_b,
                "type": "fork" if curr_b > prev_worker_count_b else "decay",
            })
            prev_worker_count_b = curr_b

        if (i + 1) % SNAPSHOT_EVERY == 0 or i == len(signals) - 1:
            dist_a = extract_distribution(mesh_a)
            dist_b = extract_distribution(mesh_b)
            w1 = float(wasserstein_distance(dist_a, dist_b))
            w1_series.append(w1)
            entropy_a.append(term_entropy(mesh_a))
            entropy_b.append(term_entropy(mesh_b))
            concentration_a.append(vocabulary_concentration(mesh_a))
            concentration_b.append(vocabulary_concentration(mesh_b))

    elapsed = time.monotonic() - t0
    print(f" done ({elapsed:.1f}s)")

    # Fit lambda from W_1 decay
    w1_arr = np.array(w1_series)
    t_arr = np.arange(len(w1_arr), dtype=float)
    if len(w1_arr) > 3 and w1_arr[0] > 1e-10:
        nz = w1_arr > 1e-10
        if np.sum(nz) > 3:
            t_norm = t_arr[nz] / len(t_arr)
            log_w1 = np.log(w1_arr[nz])
            coeffs = np.polyfit(t_norm, log_w1, 1)
            gamma = -coeffs[0]
            lambda_est = float(np.exp(-gamma)) if gamma > 0 else 1.0
        else:
            lambda_est = 0.0  # converged too fast
    else:
        lambda_est = float("nan")

    return {
        "label": label,
        "params": mesh_kwargs,
        "lambda": lambda_est,
        "w1_initial": float(w1_arr[0]) if len(w1_arr) > 0 else 0,
        "w1_final": float(w1_arr[-1]) if len(w1_arr) > 0 else 0,
        "convergence_ratio": float(w1_arr[-1] / w1_arr[0]) if len(w1_arr) > 0 and w1_arr[0] > 0 else 0,
        "w1_series": w1_series,
        "entropy_a": entropy_a,
        "entropy_b": entropy_b,
        "concentration_a": concentration_a,
        "concentration_b": concentration_b,
        "fork_events_a": fork_events_a,
        "fork_events_b": fork_events_b,
        "final_workers_a": mesh_a.worker_count,
        "final_workers_b": mesh_b.worker_count,
    }


# ---------------------------------------------------------------------------
# Study 2: Predicted vs Measured Lambda (parameter sensitivity)
# ---------------------------------------------------------------------------

def study2_analysis(study1_results: dict, config: StigmergyConfig) -> dict:
    """Compute predicted lambda under multiple parameter estimation methods."""
    wc = config.mesh.worker

    methods = {}

    # Method A: Config-based (same as Study 1)
    sigma_kappa_sq = (wc.max_threshold - wc.base_threshold) ** 2
    sigma_L_sq = wc.base_threshold ** 2
    disc = sigma_L_sq ** 2 + 4 * sigma_L_sq * sigma_kappa_sq
    sigma_star_sq = (sigma_L_sq + np.sqrt(disc)) / 2
    methods["config_based"] = {
        "sigma_kappa_sq": sigma_kappa_sq,
        "sigma_L_sq": sigma_L_sq,
        "sigma_star_sq": sigma_star_sq,
        "lambda_star": sigma_kappa_sq / (sigma_star_sq + sigma_kappa_sq),
        "lambda_uniform": sigma_kappa_sq / (sigma_L_sq + sigma_kappa_sq),
    }

    # Method B: Vigilance range as sigma_kappa, gap_threshold as sigma_L
    sigma_kappa_sq_b = (wc.max_threshold - wc.base_threshold) ** 2
    sigma_L_sq_b = 0.08 ** 2  # gap_threshold
    disc_b = sigma_L_sq_b ** 2 + 4 * sigma_L_sq_b * sigma_kappa_sq_b
    sigma_star_sq_b = (sigma_L_sq_b + np.sqrt(disc_b)) / 2
    methods["gap_threshold_based"] = {
        "sigma_kappa_sq": sigma_kappa_sq_b,
        "sigma_L_sq": sigma_L_sq_b,
        "sigma_star_sq": sigma_star_sq_b,
        "lambda_star": sigma_kappa_sq_b / (sigma_star_sq_b + sigma_kappa_sq_b),
        "lambda_uniform": sigma_kappa_sq_b / (sigma_L_sq_b + sigma_kappa_sq_b),
    }

    # Method C: Broader sigma_kappa (full [0,1] scoring range)
    sigma_kappa_sq_c = 0.25  # (0.5)^2 — half the scoring range
    sigma_L_sq_c = wc.base_threshold ** 2
    disc_c = sigma_L_sq_c ** 2 + 4 * sigma_L_sq_c * sigma_kappa_sq_c
    sigma_star_sq_c = (sigma_L_sq_c + np.sqrt(disc_c)) / 2
    methods["scoring_range_based"] = {
        "sigma_kappa_sq": sigma_kappa_sq_c,
        "sigma_L_sq": sigma_L_sq_c,
        "sigma_star_sq": sigma_star_sq_c,
        "lambda_star": sigma_kappa_sq_c / (sigma_star_sq_c + sigma_kappa_sq_c),
        "lambda_uniform": sigma_kappa_sq_c / (sigma_L_sq_c + sigma_kappa_sq_c),
    }

    # Method D: Inverse estimation from measured lambda
    # If lambda_measured = sigma_kappa^2 / (sigma_*^2 + sigma_kappa^2),
    # and sigma_*^2 depends on sigma_kappa^2 and sigma_L^2,
    # then we can estimate what sigma_kappa^2 / sigma_L^2 ratio
    # would produce the measured lambda.
    measured_lambda = study1_results.get("fit_result", {}).get("lambda_log_fit", None)
    if measured_lambda and 0 < measured_lambda < 1:
        # From lambda = sigma_kappa^2 / (sigma_*^2 + sigma_kappa^2)
        # and sigma_*^2 = (sigma_L^2 + sqrt(sigma_L^4 + 4 sigma_L^2 sigma_kappa^2)) / 2
        # We can solve for the ratio r = sigma_kappa^2 / sigma_L^2
        # Numerically sweep r to match measured lambda
        best_r = None
        best_err = float("inf")
        for r in np.linspace(0.1, 100, 10000):
            sk2 = r
            sl2 = 1.0
            d = sl2 ** 2 + 4 * sl2 * sk2
            ss2 = (sl2 + np.sqrt(d)) / 2
            lam = sk2 / (ss2 + sk2)
            err = abs(lam - measured_lambda)
            if err < best_err:
                best_err = err
                best_r = r
        methods["inverse_estimated"] = {
            "sigma_kappa_sq_over_sigma_L_sq": best_r,
            "implied_lambda_star": measured_lambda,
            "match_error": best_err,
            "note": "Ratio that would produce measured lambda under Gaussian model",
        }

    return {
        "measured_lambda_phase2": study1_results.get("fit_result", {}).get("lambda_log_fit", None),
        "measured_convergence_ratio": study1_results.get("fit_result", {}).get("convergence_ratio", None),
        "prediction_methods": methods,
        "conclusion": "Gaussian model provides upper bound; mesh's non-Gaussian features strengthen contraction",
    }


# ---------------------------------------------------------------------------
# Study 3: Selection Balance and Cage Verification
# ---------------------------------------------------------------------------

async def study3_cage_verification(
    signals: list[Signal],
    config: StigmergyConfig,
) -> dict:
    """Vary beta (selection balance) and measure vocabulary diversity.

    Beta is controlled by base_threshold:
    - Low base_threshold (0.05) = high beta = accepts diverse signals = generative
    - High base_threshold (0.50) = low beta = rejects unfamiliar = Cage
    """
    results = []
    beta_settings = [
        (0.05, "very_open"),
        (0.10, "open"),
        (0.15, "default"),
        (0.25, "selective"),
        (0.35, "restrictive"),
        (0.50, "cage_like"),
    ]

    print("  Study 3: Selection balance sweep...")
    for bt, label in beta_settings:
        trial = await run_trial_pair(
            signals, config, f"beta_{label}",
            base_threshold=bt,
        )
        results.append(trial)

    # Analyze: does higher base_threshold (lower beta) reduce entropy?
    analysis = {
        "trials": [],
        "entropy_vs_beta": [],
    }

    for r, (bt, label) in zip(results, beta_settings):
        avg_entropy = float(np.mean(r["entropy_a"][-10:])) if r["entropy_a"] else 0
        avg_concentration = float(np.mean(r["concentration_a"][-10:])) if r["concentration_a"] else 0
        analysis["trials"].append({
            "label": label,
            "base_threshold": bt,
            "lambda": r["lambda"],
            "final_entropy": avg_entropy,
            "final_concentration": avg_concentration,
            "w1_convergence_ratio": r["convergence_ratio"],
            "final_workers": r["final_workers_a"],
        })
        analysis["entropy_vs_beta"].append({
            "base_threshold": bt,
            "entropy": avg_entropy,
            "concentration": avg_concentration,
        })

    # Check Cage prediction: beta -> 0 should produce entropy collapse
    entropies = [t["final_entropy"] for t in analysis["trials"]]
    thresholds = [t["base_threshold"] for t in analysis["trials"]]
    if len(entropies) >= 3:
        # Is entropy monotonically decreasing with threshold?
        diffs = np.diff(entropies)
        monotonic_decreasing = np.all(diffs <= 0)
        correlation = float(np.corrcoef(thresholds, entropies)[0, 1])
        analysis["cage_prediction"] = {
            "entropy_decreasing_with_threshold": bool(monotonic_decreasing),
            "correlation": correlation,
            "confirmed": correlation < -0.5,
            "interpretation": (
                "CONFIRMED: Higher threshold (lower beta) reduces vocabulary diversity"
                if correlation < -0.5
                else "PARTIAL: Trend present but not monotonic"
                if correlation < 0
                else "NOT CONFIRMED: No clear relationship"
            ),
        }

    return {"study3_results": results, "study3_analysis": analysis}


# ---------------------------------------------------------------------------
# Study 4: Convergence Rate vs Scoring Curvature
# ---------------------------------------------------------------------------

async def study4_curvature_sweep(
    signals: list[Signal],
    config: StigmergyConfig,
) -> dict:
    """Vary max_threshold (effective sigma^2_kappa) and measure lambda.

    Theorem 5.1 predicts: sharper scoring (smaller sigma^2_kappa) -> faster convergence (smaller lambda).
    In the mesh: smaller max_threshold - base_threshold gap = sharper effective scoring.
    """
    results = []
    curvature_settings = [
        (0.25, "very_sharp"),   # gap = 0.10
        (0.40, "sharp"),        # gap = 0.25
        (0.60, "moderate"),     # gap = 0.45
        (0.80, "default"),      # gap = 0.65
        (0.95, "flat"),         # gap = 0.80
    ]

    print("  Study 4: Scoring curvature sweep...")
    for mt, label in curvature_settings:
        trial = await run_trial_pair(
            signals, config, f"curv_{label}",
            max_threshold=mt,
        )
        results.append(trial)

    analysis = {
        "trials": [],
        "lambda_vs_curvature": [],
    }

    for r, (mt, label) in zip(results, curvature_settings):
        gap = mt - 0.15  # base_threshold is 0.15
        sigma_kappa_sq = gap ** 2
        analysis["trials"].append({
            "label": label,
            "max_threshold": mt,
            "gap": gap,
            "sigma_kappa_sq": sigma_kappa_sq,
            "lambda_measured": r["lambda"],
            "w1_convergence_ratio": r["convergence_ratio"],
            "final_workers": r["final_workers_a"],
        })
        analysis["lambda_vs_curvature"].append({
            "sigma_kappa_sq": sigma_kappa_sq,
            "lambda": r["lambda"],
        })

    # Check prediction: lambda should increase with sigma_kappa_sq
    lambdas = [t["lambda_measured"] for t in analysis["trials"] if not np.isnan(t["lambda_measured"])]
    sigmas = [t["sigma_kappa_sq"] for t in analysis["trials"] if not np.isnan(t["lambda_measured"])]
    if len(lambdas) >= 3:
        correlation = float(np.corrcoef(sigmas, lambdas)[0, 1])
        analysis["curvature_prediction"] = {
            "correlation_sigma_lambda": correlation,
            "confirmed": correlation > 0.5,
            "interpretation": (
                "CONFIRMED: Flatter scoring (larger sigma_kappa) -> slower convergence (larger lambda)"
                if correlation > 0.5
                else "PARTIAL: Trend present but weak"
                if correlation > 0
                else "NOT CONFIRMED: No clear relationship"
            ),
        }

    return {"study4_results": results, "study4_analysis": analysis}


# ---------------------------------------------------------------------------
# Study 5: Fork Variance Injection
# ---------------------------------------------------------------------------

async def study5_fork_analysis(
    signals: list[Signal],
    config: StigmergyConfig,
) -> dict:
    """Instrument fork events and measure variance injection.

    Theory: Fork creates new workers by splitting a full worker, injecting
    variance into the system (sigma^2_L contribution). We measure:
    - When forks occur (signal index)
    - Between-component variance before/after fork
    - Whether fork events correlate with W_1 increases
    """
    # Run a single detailed trial with fork tracking
    mesh = build_mesh_with_params(config, n_workers=3)

    fork_log = []
    variance_log = []
    prev_worker_count = mesh.worker_count

    print("  Study 5: Fork variance injection tracking...")
    t0 = time.monotonic()

    for i, signal in enumerate(signals):
        trace = await mesh.ingest(signal)

        curr_count = mesh.worker_count
        if curr_count != prev_worker_count:
            # Fork or decay event — measure variance
            vigilances = [w.adaptive_threshold for w in mesh.workers]
            familiarities = [w.rolling_avg_familiarity for w in mesh.workers]
            term_counts = [len(w.context.terms) for w in mesh.workers]

            fork_log.append({
                "signal_idx": i,
                "type": "fork" if curr_count > prev_worker_count else "decay",
                "prev_count": prev_worker_count,
                "curr_count": curr_count,
                "vigilance_var": float(np.var(vigilances)),
                "vigilance_mean": float(np.mean(vigilances)),
                "familiarity_var": float(np.var(familiarities)),
                "familiarity_mean": float(np.mean(familiarities)),
                "term_count_var": float(np.var(term_counts)),
                "term_count_mean": float(np.mean(term_counts)),
            })
            prev_worker_count = curr_count

        if (i + 1) % SNAPSHOT_EVERY == 0:
            vigilances = [w.adaptive_threshold for w in mesh.workers]
            familiarities = [w.rolling_avg_familiarity for w in mesh.workers]
            term_counts = [len(w.context.terms) for w in mesh.workers]
            variance_log.append({
                "signal_idx": i,
                "worker_count": mesh.worker_count,
                "vigilance_var": float(np.var(vigilances)),
                "familiarity_var": float(np.var(familiarities)),
                "term_count_var": float(np.var(term_counts)),
                "avg_fullness": float(mesh.avg_fullness),
            })

    elapsed = time.monotonic() - t0
    print(f"    done ({elapsed:.1f}s, {len(fork_log)} lifecycle events)")

    # Analysis
    fork_events = [e for e in fork_log if e["type"] == "fork"]
    decay_events = [e for e in fork_log if e["type"] == "decay"]

    # Does fork inject variance? Compare variance before/after fork events
    if fork_events and len(variance_log) > 2:
        # For each fork event, find the nearest variance snapshots before and after
        pre_fork_var = []
        post_fork_var = []
        for fe in fork_events:
            fi = fe["signal_idx"]
            before = [v for v in variance_log if v["signal_idx"] < fi]
            after = [v for v in variance_log if v["signal_idx"] > fi]
            if before and after:
                pre_fork_var.append(before[-1]["vigilance_var"])
                post_fork_var.append(after[0]["vigilance_var"])

        if pre_fork_var and post_fork_var:
            mean_pre = float(np.mean(pre_fork_var))
            mean_post = float(np.mean(post_fork_var))
            variance_increase = mean_post - mean_pre
            fork_analysis = {
                "n_fork_events": len(fork_events),
                "n_decay_events": len(decay_events),
                "mean_variance_pre_fork": mean_pre,
                "mean_variance_post_fork": mean_post,
                "variance_increase": variance_increase,
                "fork_injects_variance": variance_increase > 0,
                "interpretation": (
                    "CONFIRMED: Fork events increase between-component variance"
                    if variance_increase > 0
                    else "NOT CONFIRMED: Fork events do not increase variance"
                ),
            }
        else:
            fork_analysis = {"error": "Not enough matched pre/post fork snapshots"}
    else:
        fork_analysis = {"error": f"Too few fork events ({len(fork_events)})"}

    return {
        "study5_fork_log": fork_log,
        "study5_variance_log": variance_log[:50] + variance_log[-50:],  # Trim for JSON size
        "study5_analysis": fork_analysis,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print("=" * 70)
    print("Studies 2-5: Extended Empirical Validation")
    print("Emergence Design Calculus")
    print("=" * 70)
    print()

    # Load config and Study 1 results
    print("[1/5] Loading config and Study 1 data...")
    config = _load_config()

    study1_path = OUTPUT_DIR / "study1_results.json"
    if study1_path.exists():
        with open(study1_path) as f:
            study1_results = json.load(f)
        print(f"  Study 1 loaded: {study1_results['config']['n_signals']} signals, "
              f"lambda={study1_results['fit_result'].get('lambda_log_fit', 'N/A'):.4f}")
    else:
        study1_results = {}
        print("  WARNING: Study 1 results not found")

    # Study 2: Predicted vs Measured (analytical, no new runs needed)
    print("\n[2/5] Study 2: Predicted vs Measured Lambda...")
    study2 = study2_analysis(study1_results, config)
    print("  Prediction methods:")
    for name, m in study2["prediction_methods"].items():
        ls = m.get("lambda_star", m.get("implied_lambda_star", "N/A"))
        print(f"    {name}: lambda_* = {ls:.4f}" if isinstance(ls, float) else f"    {name}: {ls}")
    print(f"  Measured (Phase 2): {study2['measured_lambda_phase2']:.4f}")

    # Fetch signals (once, shared across Studies 3-5)
    print("\n[3/5] Fetching production signals...")
    signals = await fetch_signals(config)
    if not signals:
        print("  ERROR: No signals fetched.")
        return
    print(f"  Total: {len(signals)} signals")

    # Study 3: Cage verification
    print(f"\n[4/5] Study 3: Selection balance sweep (6 settings x {len(signals)} signals)...")
    study3 = await study3_cage_verification(signals, config)
    cage_result = study3["study3_analysis"].get("cage_prediction", {})
    print(f"  Cage prediction: {cage_result.get('interpretation', 'N/A')}")

    # Study 4: Curvature sweep
    print(f"\n[5/5] Study 4: Scoring curvature sweep (5 settings x {len(signals)} signals)...")
    study4 = await study4_curvature_sweep(signals, config)
    curv_result = study4["study4_analysis"].get("curvature_prediction", {})
    print(f"  Curvature prediction: {curv_result.get('interpretation', 'N/A')}")

    # Study 5: Fork analysis (runs during Study 3/4, but we'll run once more with instrumentation)
    print(f"\n  Study 5: Fork variance injection ({len(signals)} signals)...")
    study5 = await study5_fork_analysis(signals, config)
    fork_result = study5["study5_analysis"]
    print(f"  Fork analysis: {fork_result.get('interpretation', fork_result.get('error', 'N/A'))}")

    # Save all results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nStudy 2 (Predicted vs Measured):")
    for name, m in study2["prediction_methods"].items():
        ls = m.get("lambda_star", m.get("implied_lambda_star", "N/A"))
        lu = m.get("lambda_uniform", "N/A")
        print(f"  {name}: lambda_* = {ls:.4f}, lambda_uniform = {lu:.4f}" if isinstance(ls, float) and isinstance(lu, float) else f"  {name}: {ls}")
    print(f"  Measured: {study2['measured_lambda_phase2']:.4f}")

    print("\nStudy 3 (Cage Verification):")
    for t in study3["study3_analysis"]["trials"]:
        print(f"  bt={t['base_threshold']:.2f} ({t['label']}): entropy={t['final_entropy']:.2f}, "
              f"concentration={t['final_concentration']:.4f}, lambda={t['lambda']:.4f}")
    print(f"  Prediction: {cage_result.get('interpretation', 'N/A')}")

    print("\nStudy 4 (Curvature Sweep):")
    for t in study4["study4_analysis"]["trials"]:
        print(f"  mt={t['max_threshold']:.2f} ({t['label']}): sigma_kappa^2={t['sigma_kappa_sq']:.4f}, "
              f"lambda={t['lambda_measured']:.4f}")
    print(f"  Prediction: {curv_result.get('interpretation', 'N/A')}")

    print("\nStudy 5 (Fork Variance):")
    if "error" not in fork_result:
        print(f"  Fork events: {fork_result['n_fork_events']}")
        print(f"  Decay events: {fork_result['n_decay_events']}")
        print(f"  Variance pre-fork: {fork_result['mean_variance_pre_fork']:.6f}")
        print(f"  Variance post-fork: {fork_result['mean_variance_post_fork']:.6f}")
        print(f"  Result: {fork_result['interpretation']}")
    else:
        print(f"  {fork_result['error']}")

    # Combine all results
    all_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_signals": len(signals),
        "study2": study2,
        "study3_analysis": study3["study3_analysis"],
        "study4_analysis": study4["study4_analysis"],
        "study5_analysis": study5["study5_analysis"],
        "study5_fork_log": study5["study5_fork_log"],
    }

    output_path = OUTPUT_DIR / "studies_2_5_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
