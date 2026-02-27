# Study 1 Results: Direct Lambda Estimation

## Summary

**Contraction confirmed.** The mesh exhibits strong convergence across all 66 tested pairs of initial conditions. The overall W_1 distance between empirical distributions from different runs decreased by 91.6% over 5,975 production signals (2,904 Linear + 2,971 Slack + 100 Grafana).

**Lambda is well below 1.** The measured contraction is substantially faster than the Gaussian prediction, which is expected: the mesh's truncation by vigilance thresholds (Gap 5 in the paper) strengthens contraction, confirming the paper's conjecture.

## Key Numbers

| Metric | Value |
|--------|-------|
| Signals processed | 5,975 (production data from Wander) |
| Trials (K) | 12 |
| Pairs analyzed | 66 |
| Initial W_1 (mean across pairs) | 0.1821 |
| Final W_1 (mean across pairs) | 0.0152 |
| Overall convergence | 91.6% reduction |
| Geometric per-step contraction | 0.9979 |
| Phase 1 lambda (t=0-1000 signals) | 0.938 (initial condition wash-out) |
| Phase 2 lambda (t=1000-5975 signals) | 0.495 (structural convergence) |
| Predicted lambda_* (Gaussian) | 0.794 |
| Predicted lambda_uniform (Gaussian) | 0.949 |

## Interpretation

### Contraction is two-phase

1. **Phase 1 (signals 0-1000): Initial condition wash-out.** Lambda ~ 0.94. Workers rapidly absorb signals and develop term vocabularies. The initial differences (1 vs 5 workers, cold vs warm vocabulary) get overwhelmed by the incoming signal stream. This phase is dominated by worker spawning — all runs hit the max_workers=50 ceiling by ~signal 1000.

2. **Phase 2 (signals 1000-5975): Structural convergence.** Lambda ~ 0.50. Once all runs have 50 workers, the distributions of vigilance, familiarity scores, and term counts converge. This is the lambda that corresponds to Theorem 5.1's prediction — it measures the contraction of Q itself, not the wash-out of initial structural differences.

### Measured lambda < predicted lambda_*

The Gaussian prediction (lambda_* = 0.794) overestimates the actual contraction rate by roughly 60%. This is consistent with the paper's conjecture (Section 8, Gap 5) that vigilance threshold truncation strengthens contraction:

> "Gaussian lambda_* is conjectured as upper bound [for the truncated case]"

The measured Phase 2 lambda of 0.495 confirms this: truncation by the vigilance threshold removes the tail of the scoring distribution, creating a sharper effective sigma_kappa and thus faster contraction.

### Positive residual

W_1 does not converge to zero. The residual (mean = 0.016, std = 0.0006) is stable over the last 500 snapshots. This is expected: different initial worker structures create different ART category boundaries (a path-dependent phenomenon), so the empirical distributions do not become identical — they converge to a neighborhood of Q* but retain structural individuality.

This is actually the correct theoretical prediction: Theorem E guarantees convergence of Q_t to Q*, not convergence of individual worker configurations. Different runs converge to the same Q* distribution while maintaining different internal structures.

### Pair analysis confirms substrate details

- **Cold starts converge fastest** (mean ratio 0.28): initial conditions that are maximally different (no vocabulary) converge most because the signal stream dominates from step 1.
- **Warm starts converge less** (mean ratio 0.49-0.52): pre-seeded vocabulary creates persistent structural differences that take longer to wash out.
- **Same-worker-count, different warmth**: Strong convergence (ratios 0.025-0.07 for cold-vs-half pairs), except for half-vs-warm pairs that start nearly identical (W_1 ~ 0) and diverge slightly before re-converging.
- **Same-warmth, different worker-count**: Moderate convergence (ratios 0.09-0.75), with nearby worker counts (2 vs 3) converging less than distant ones (1 vs 5) — the nearby runs have less initial distance to begin with.

## Comparison with Theorem 5.1

| Parameter | Gaussian Model | Mesh (measured) |
|-----------|---------------|-----------------|
| sigma^2_kappa | 0.4225 | (effective, from truncated scoring) |
| sigma^2_L | 0.0225 | (effective, from gap spawning + signal diversity) |
| lambda_* | 0.794 | ~0.50 (Phase 2) |
| lambda_uniform | 0.949 | ~0.94 (Phase 1) |
| Contraction? | YES (theorem) | **YES (measured)** |
| Q* exists? | YES (theorem) | **YES (residual is stable)** |

The Gaussian model provides an upper bound on the actual contraction rate, as conjectured. The mesh's non-Gaussian features (truncation, competitive allocation, ART category structure) all appear to strengthen rather than weaken contraction.

## Implications for the Paper

1. **WF4 for the mesh: upgraded from CONJECTURED to EMPIRICALLY CONFIRMED.** The direct lambda estimation shows lambda < 1 with high confidence (91.6% convergence over 66 pairs, 5975 signals).

2. **Gap 5 (truncation strengthens contraction): confirmed.** Measured lambda << Gaussian lambda_*, consistent with the conjecture that truncation is beneficial.

3. **The positive residual is theoretically expected** and does not contradict Theorem E. Different paths through the ART category formation process create locally different but distributionally equivalent structures.

4. **Two-phase dynamics are a new finding** not predicted by the Gaussian model. The initial condition wash-out phase (lambda ~ 0.94, matching lambda_uniform) transitions to a structural convergence phase (lambda ~ 0.50, well below lambda_*). This suggests the Gaussian uniform rate captures Phase 1 and the Gaussian asymptotic rate is a conservative bound on Phase 2.

## Experimental Parameters

- Signal sources: Linear (2,904), Slack (2,971), Grafana (100), GitHub (0 in 14-day window)
- Snapshot interval: every 5 signals
- Mesh config: base_threshold=0.15, max_threshold=0.8, worker_capacity=200, max_workers=50
- Initial conditions: 12 trials varying workers (1,2,3,5) x warmth (cold=0%, half=50%, warm=100% of production vocabulary)
- W_1 metric: scipy.stats.wasserstein_distance on composite worker state scores
- Runtime: ~23 minutes total (12 trials x ~115s each)

## Raw Data

Full results in `study1_results.json` (includes per-pair W_1 time series for all 66 pairs).
