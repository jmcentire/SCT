#!/usr/bin/env python3
"""
Phase 2c (GPT-2 124M) Analysis Script
Mirrors Phase 3 (Qwen 1.5B) analysis format exactly.
"""

import json
import numpy as np
from collections import defaultdict

# Load data
seeds = [42, 43]
data = {}
for seed in seeds:
    path = f"/Users/jmcentire/Code/AI/results/phase2c/seed{seed}/combined_results.json"
    with open(path) as f:
        data[seed] = json.load(f)

print("=" * 80)
print("PHASE 2c ANALYSIS: GPT-2 124M on WikiText-103 (2000 steps)")
print("Seeds: 42, 43")
print("=" * 80)

# ============================================================================
# 1. REGIME BREAKDOWN
# ============================================================================
print("\n" + "=" * 80)
print("1. REGIME BREAKDOWN")
print("=" * 80)

regime_counts = {}
for seed in seeds:
    rt = data[seed]["regime_timeline"]
    counts = defaultdict(int)
    for regime in rt["regimes"]:
        counts[regime] += 1

    regime_counts[seed] = dict(counts)
    total = len(rt["regimes"])
    print(f"\nSeed {seed}: total checkpoints = {total}")
    for regime in ["chaotic", "transition", "stable", "unknown"]:
        c = counts.get(regime, 0)
        print(f"  {regime:12s}: {c:3d} ({100*c/total:.1f}%)")

# Mean +/- std across seeds
print("\nMean +/- Std across seeds:")
for regime in ["chaotic", "transition", "stable", "unknown"]:
    vals = [regime_counts[s].get(regime, 0) for s in seeds]
    print(f"  {regime:12s}: {np.mean(vals):.1f} +/- {np.std(vals):.1f}")

# ============================================================================
# 2. K-SWEEP STRICT ACCEPTANCE RATES
# ============================================================================
print("\n" + "=" * 80)
print("2. K-SWEEP STRICT ACCEPTANCE RATES (by regime)")
print("=" * 80)

K_values = [5, 10, 25, 50, 75, 100]
predictors = ["momentum", "linear", "quadratic"]

for regime_filter in ["transition", "stable"]:
    print(f"\n--- Regime: {regime_filter} ---")
    print(f"{'K':>5s} | {'momentum':>20s} | {'linear':>20s} | {'quadratic':>20s}")
    print("-" * 75)

    for K in K_values:
        parts = []
        for pred in predictors:
            rates = []
            for seed in seeds:
                evals = data[seed]["depth_sweep"]["evaluations"]
                matching = [e for e in evals
                           if e["K"] == K and e["predictor"] == pred and e["regime"] == regime_filter]
                if matching:
                    accepted = sum(1 for e in matching if e.get("accepted_strict", False))
                    rate = accepted / len(matching)
                    rates.append(rate)

            if rates:
                mean_rate = np.mean(rates)
                std_rate = np.std(rates)
                parts.append(f"{mean_rate*100:6.1f}% +/- {std_rate*100:5.1f}%")
            else:
                parts.append(f"{'N/A':>20s}")

        print(f"{K:5d} | {parts[0]:>20s} | {parts[1]:>20s} | {parts[2]:>20s}")

# Also print the raw per-seed values from summary_by_k for reference
print("\n  [Per-seed detail from summary_by_k]")
for seed in seeds:
    sbk = data[seed]["depth_sweep"]["summary_by_k"]
    print(f"\n  Seed {seed}:")
    print(f"  {'K':>5s} | {'Regime':>12s} | {'n':>4s} | {'momentum_strict':>16s} | {'linear_strict':>16s} | {'quadratic_strict':>16s}")
    print("  " + "-" * 80)
    for K in K_values:
        k_data = sbk.get(str(K), {})
        for regime in ["transition", "stable"]:
            row_parts = []
            for pred in predictors:
                pred_data = k_data.get(pred, {}).get(regime, None)
                if pred_data:
                    row_parts.append(f"{pred_data['strict_rate']*100:6.1f}% (n={pred_data['n']})")
                else:
                    row_parts.append(f"{'N/A':>16s}")
            n_val = k_data.get(predictors[0], {}).get(regime, {}).get("n", "-")
            print(f"  {K:5d} | {regime:>12s} | {str(n_val):>4s} | {row_parts[0]:>16s} | {row_parts[1]:>16s} | {row_parts[2]:>16s}")

# ============================================================================
# 3. K-SWEEP PCT/PROXIMITY ACCEPTANCE RATES
# ============================================================================
print("\n" + "=" * 80)
print("3. K-SWEEP PCT/PROXIMITY ACCEPTANCE RATES (by regime)")
print("=" * 80)

for regime_filter in ["transition", "stable"]:
    print(f"\n--- Regime: {regime_filter} ---")
    print(f"{'K':>5s} | {'momentum':>20s} | {'linear':>20s} | {'quadratic':>20s}")
    print("-" * 75)

    for K in K_values:
        parts = []
        for pred in predictors:
            rates = []
            for seed in seeds:
                evals = data[seed]["depth_sweep"]["evaluations"]
                matching = [e for e in evals
                           if e["K"] == K and e["predictor"] == pred and e["regime"] == regime_filter]
                if matching:
                    accepted = sum(1 for e in matching if e.get("accepted_pct", False))
                    rate = accepted / len(matching)
                    rates.append(rate)

            if rates:
                mean_rate = np.mean(rates)
                std_rate = np.std(rates)
                parts.append(f"{mean_rate*100:6.1f}% +/- {std_rate*100:5.1f}%")
            else:
                parts.append(f"{'N/A':>20s}")

        print(f"{K:5d} | {parts[0]:>20s} | {parts[1]:>20s} | {parts[2]:>20s}")

# Per-seed pct rates from summary_by_k
print("\n  [Per-seed pct_rate detail from summary_by_k]")
for seed in seeds:
    sbk = data[seed]["depth_sweep"]["summary_by_k"]
    print(f"\n  Seed {seed}:")
    print(f"  {'K':>5s} | {'Regime':>12s} | {'momentum_pct':>14s} | {'linear_pct':>14s} | {'quadratic_pct':>14s}")
    print("  " + "-" * 70)
    for K in K_values:
        k_data = sbk.get(str(K), {})
        for regime in ["transition", "stable"]:
            row_parts = []
            for pred in predictors:
                pred_data = k_data.get(pred, {}).get(regime, None)
                if pred_data:
                    row_parts.append(f"{pred_data['pct_rate']*100:6.1f}%")
                else:
                    row_parts.append(f"{'N/A':>14s}")
            print(f"  {K:5d} | {regime:>12s} | {row_parts[0]:>14s} | {row_parts[1]:>14s} | {row_parts[2]:>14s}")

# ============================================================================
# 4. MOMENTUM CATASTROPHE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. MOMENTUM CATASTROPHE ANALYSIS")
print("=" * 80)
print("(For momentum predictor only, examining predicted vs actual loss)")

print(f"\n{'K':>5s} | {'Mean Predicted':>16s} | {'Mean Actual':>16s} | {'Ratio (P/A)':>14s} | {'Catastrophe?':>14s}")
print("-" * 80)

for K in K_values:
    predicted_all = []
    actual_all = []
    for seed in seeds:
        evals = data[seed]["depth_sweep"]["evaluations"]
        matching = [e for e in evals if e["K"] == K and e["predictor"] == "momentum"]
        for e in matching:
            predicted_all.append(e["predicted_loss"])
            actual_all.append(e["current_loss"])

    mean_pred = np.mean(predicted_all)
    mean_actual = np.mean(actual_all)
    ratio = mean_pred / mean_actual
    catastrophe = "YES" if ratio > 10 else ("MILD" if ratio > 2 else "NO")

    print(f"{K:5d} | {mean_pred:16.4f} | {mean_actual:16.4f} | {ratio:14.4f} | {catastrophe:>14s}")

# Break down by regime
print("\n--- Momentum Predicted/Actual by Regime ---")
for regime_filter in ["unknown", "chaotic", "transition", "stable"]:
    has_data = False
    for seed in seeds:
        evals = data[seed]["depth_sweep"]["evaluations"]
        matching = [e for e in evals if e["predictor"] == "momentum" and e["regime"] == regime_filter]
        if matching:
            has_data = True
            break

    if not has_data:
        continue

    print(f"\nRegime: {regime_filter}")
    print(f"{'K':>5s} | {'Mean Predicted':>16s} | {'Mean Actual':>16s} | {'Ratio (P/A)':>14s} | {'Max Predicted':>14s}")
    print("-" * 75)

    for K in K_values:
        predicted_all = []
        actual_all = []
        for seed in seeds:
            evals = data[seed]["depth_sweep"]["evaluations"]
            matching = [e for e in evals
                       if e["K"] == K and e["predictor"] == "momentum" and e["regime"] == regime_filter]
            for e in matching:
                predicted_all.append(e["predicted_loss"])
                actual_all.append(e["current_loss"])

        if predicted_all:
            mean_pred = np.mean(predicted_all)
            mean_actual = np.mean(actual_all)
            max_pred = np.max(predicted_all)
            ratio = mean_pred / mean_actual
            print(f"{K:5d} | {mean_pred:16.4f} | {mean_actual:16.4f} | {ratio:14.4f} | {max_pred:14.4f}")

# ============================================================================
# 5. TRAINING SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("5. TRAINING SUMMARY")
print("=" * 80)

for seed in seeds:
    s = data[seed]["summary"]
    c = data[seed]["config"]
    print(f"\nSeed {seed}:")
    print(f"  Phase0 baseline loss: {c.get('phase0_baseline_loss', 'N/A')}")
    print(f"  Total steps trained:  {s['total_steps_trained']}")
    print(f"  Total steps skipped:  {s['total_steps_skipped']}")
    print(f"  Final val loss:       {s['final_val_loss']:.6f}")
    print(f"  Pass 1 time:          {s['pass1_time']:.2f}s ({s['pass1_time']/60:.1f} min)")
    print(f"  Pass 2 time:          {s['pass2_time']:.2f}s ({s['pass2_time']/60:.1f} min)")
    print(f"  Pass 3 time:          {s['pass3_time']:.2f}s ({s['pass3_time']/60:.1f} min)")
    total_time = s['pass1_time'] + s['pass2_time'] + s['pass3_time']
    print(f"  Total time:           {total_time:.2f}s ({total_time/60:.1f} min)")

# Mean across seeds
final_losses = [data[s]["summary"]["final_val_loss"] for s in seeds]
pass1_times = [data[s]["summary"]["pass1_time"] for s in seeds]
pass2_times = [data[s]["summary"]["pass2_time"] for s in seeds]
pass3_times = [data[s]["summary"]["pass3_time"] for s in seeds]
total_times = [p1+p2+p3 for p1,p2,p3 in zip(pass1_times, pass2_times, pass3_times)]

print(f"\nAcross seeds (mean +/- std):")
print(f"  Final val loss:  {np.mean(final_losses):.6f} +/- {np.std(final_losses):.6f}")
print(f"  Pass 1 time:     {np.mean(pass1_times):.2f}s +/- {np.std(pass1_times):.2f}s")
print(f"  Pass 2 time:     {np.mean(pass2_times):.2f}s +/- {np.std(pass2_times):.2f}s")
print(f"  Pass 3 time:     {np.mean(pass3_times):.2f}s +/- {np.std(pass3_times):.2f}s")
print(f"  Total time:      {np.mean(total_times):.2f}s +/- {np.std(total_times):.2f}s ({np.mean(total_times)/60:.1f} min)")

# ============================================================================
# 6. CASCADE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("6. CASCADE RESULTS")
print("=" * 80)

# Group all cascade results by config type across seeds
cascade_summary = defaultdict(lambda: defaultdict(list))

for seed in seeds:
    cascade = data[seed].get("cascade_results", {})
    configs = cascade.get("configurations", [])
    cascade_time = cascade.get("time_seconds", "N/A")
    print(f"\nSeed {seed}: {len(configs)} cascade configurations, time={cascade_time}s")

    # Group by (chain_length, K_per_link)
    config_groups = defaultdict(list)
    for cfg in configs:
        key = (cfg["chain_length"], cfg["K_per_link"])
        config_groups[key].append(cfg)

    for (chain_len, k_per), cfgs in sorted(config_groups.items()):
        print(f"\n  Config: chain_length={chain_len}, K_per_link={k_per}, total_depth={chain_len*k_per}")
        print(f"  Number of cascade runs: {len(cfgs)}")

        for cfg in cfgs:
            checkpoint = cfg["checkpoint_step"]
            links = cfg["links"]
            initial_loss = links[0]["predicted_loss"]
            final_loss = links[-1]["trained_loss"]
            total_drift = sum(l["l2_drift"] for l in links)
            max_drift = max(l["l2_drift"] for l in links)
            loss_delta = final_loss - initial_loss

            # Track per-link drifts
            link_drifts = [l["l2_drift"] for l in links]
            link_loss_deltas = [l["trained_loss"] - l["predicted_loss"] for l in links]

            print(f"    Checkpoint {checkpoint}:")
            print(f"      initial_loss={initial_loss:.6f}, final_loss={final_loss:.6f}, loss_delta={loss_delta:+.6f}")
            print(f"      total_l2_drift={total_drift:.6f}, max_link_drift={max_drift:.6f}")
            print(f"      per-link drifts: {['%.6f' % d for d in link_drifts]}")

            # Store for cross-seed summary
            cascade_summary[(chain_len, k_per)][seed].append({
                "checkpoint": checkpoint,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_delta": loss_delta,
                "total_drift": total_drift,
                "max_drift": max_drift,
            })

# Cross-seed cascade summary
print("\n--- Cross-seed Cascade Summary ---")
for (chain_len, k_per) in sorted(cascade_summary.keys()):
    print(f"\nConfig: chain_length={chain_len}, K_per_link={k_per}, total_depth={chain_len*k_per}")
    all_deltas = []
    all_drifts = []
    # Also separate "good" (checkpoint 1355 = stable regime) vs others
    stable_deltas = []
    stable_drifts = []
    for seed in seeds:
        for entry in cascade_summary[(chain_len, k_per)].get(seed, []):
            if not np.isnan(entry["loss_delta"]) and not np.isnan(entry["total_drift"]):
                all_deltas.append(entry["loss_delta"])
                all_drifts.append(entry["total_drift"])
                if entry["checkpoint"] == 1355:
                    stable_deltas.append(entry["loss_delta"])
                    stable_drifts.append(entry["total_drift"])
    if all_deltas:
        print(f"  ALL checkpoints (excluding NaN):")
        print(f"    Loss delta: mean={np.mean(all_deltas):+.6f}, std={np.std(all_deltas):.6f}, n={len(all_deltas)}")
        print(f"    L2 drift:   mean={np.mean(all_drifts):.6f}, std={np.std(all_drifts):.6f}")
    if stable_deltas:
        print(f"  STABLE checkpoint (step 1355) only:")
        print(f"    Loss delta: mean={np.mean(stable_deltas):+.6f}, std={np.std(stable_deltas):.6f}, n={len(stable_deltas)}")
        print(f"    L2 drift:   mean={np.mean(stable_drifts):.6f}, std={np.std(stable_drifts):.6f}")

# ============================================================================
# 7. PREDICTION LOG ANALYSIS (Pass 1 - momentum only)
# ============================================================================
print("\n" + "=" * 80)
print("7. PREDICTION LOG ANALYSIS (Pass 1 - Momentum)")
print("=" * 80)

for seed in seeds:
    pred_log = data[seed].get("prediction_log", [])
    if not pred_log:
        print(f"\nSeed {seed}: No prediction log found.")
        continue

    total = len(pred_log)
    accepted_loss = sum(1 for e in pred_log if e.get("accepted_by_loss", False))
    applied = sum(1 for e in pred_log if e.get("actually_applied", False))

    print(f"\nSeed {seed}:")
    print(f"  Total predictions: {total}")
    print(f"  Accepted by loss: {accepted_loss} ({100*accepted_loss/total:.1f}%)")
    print(f"  Actually applied: {applied} ({100*applied/total:.1f}%)")

    # Breakdown by K
    k_groups = defaultdict(list)
    for e in pred_log:
        k_groups[e["K"]].append(e)

    print(f"\n  {'K':>5s} | {'Total':>6s} | {'Accept Loss':>12s} | {'Applied':>10s} | {'Mean Sim':>10s} | {'Mean Delta':>12s}")
    print("  " + "-" * 70)
    for K in sorted(k_groups.keys()):
        entries = k_groups[K]
        n = len(entries)
        acc = sum(1 for e in entries if e.get("accepted_by_loss", False))
        app = sum(1 for e in entries if e.get("actually_applied", False))
        sims = [e["similarity"] for e in entries if "similarity" in e]
        deltas = [e["delta"] for e in entries]
        mean_sim = np.mean(sims) if sims else float('nan')
        mean_delta = np.mean(deltas) if deltas else float('nan')
        print(f"  {K:5d} | {n:6d} | {acc:4d} ({100*acc/n:5.1f}%) | {app:3d} ({100*app/n:5.1f}%) | {mean_sim:.6f} | {mean_delta:12.6f}")

# ============================================================================
# 8. ADAPTIVE ACCEPTANCE RATES (K-Sweep)
# ============================================================================
print("\n" + "=" * 80)
print("8. ADAPTIVE ACCEPTANCE RATES (K-Sweep)")
print("=" * 80)

for regime_filter in ["transition", "stable"]:
    print(f"\n--- Regime: {regime_filter} ---")
    print(f"{'K':>5s} | {'momentum':>20s} | {'linear':>20s} | {'quadratic':>20s}")
    print("-" * 75)

    for K in K_values:
        parts = []
        for pred in predictors:
            rates = []
            for seed in seeds:
                evals = data[seed]["depth_sweep"]["evaluations"]
                matching = [e for e in evals
                           if e["K"] == K and e["predictor"] == pred and e["regime"] == regime_filter]
                if matching:
                    accepted = sum(1 for e in matching if e.get("accepted_adaptive", False))
                    rate = accepted / len(matching)
                    rates.append(rate)

            if rates:
                mean_rate = np.mean(rates)
                std_rate = np.std(rates)
                parts.append(f"{mean_rate*100:6.1f}% +/- {std_rate*100:5.1f}%")
            else:
                parts.append(f"{'N/A':>20s}")

        print(f"{K:5d} | {parts[0]:>20s} | {parts[1]:>20s} | {parts[2]:>20s}")

# ============================================================================
# 9. LANDSCAPE PROBE V2 SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("9. LANDSCAPE PROBE V2 SUMMARY")
print("=" * 80)

for seed in seeds:
    probes = data[seed].get("landscape_probe_v2", [])
    if not probes:
        print(f"\nSeed {seed}: No landscape probe data found.")
        continue

    print(f"\nSeed {seed}: {len(probes)} probe entries")

    # Group by regime
    regime_probes = defaultdict(list)
    for p in probes:
        regime_probes[p.get("regime", "unknown")].append(p)

    for regime in ["transition", "stable", "unknown", "chaotic"]:
        entries = regime_probes.get(regime, [])
        if not entries:
            continue

        curvatures = [e["curvature_estimate"] for e in entries if e.get("curvature_estimate") is not None]
        dir_derivs = [e["directional_derivative"] for e in entries if e.get("directional_derivative") is not None]
        dir_norms = [e["direction_norm"] for e in entries if e.get("direction_norm") is not None]

        print(f"\n  Regime: {regime} (n={len(entries)})")
        if curvatures:
            print(f"    Curvature estimate: mean={np.mean(curvatures):.6f}, std={np.std(curvatures):.6f}")
        if dir_derivs:
            print(f"    Directional deriv:  mean={np.mean(dir_derivs):.6f}, std={np.std(dir_derivs):.6f}")
        if dir_norms:
            print(f"    Direction norm:     mean={np.mean(dir_norms):.4f}, std={np.std(dir_norms):.4f}")

# ============================================================================
# 10. KEY COMPARISON METRICS (GPT-2 124M vs Qwen 1.5B)
# ============================================================================
print("\n" + "=" * 80)
print("10. KEY METRICS FOR PAPER COMPARISON (GPT-2 124M)")
print("=" * 80)

# Stable regime, strict acceptance - the main paper metric
print("\nSTRICT ACCEPTANCE RATES - STABLE REGIME (mean +/- std):")
print(f"{'K':>5s} | {'momentum':>20s} | {'linear':>20s} | {'quadratic':>20s}")
print("-" * 75)
for K in K_values:
    parts = []
    for pred in predictors:
        rates = []
        for seed in seeds:
            evals = data[seed]["depth_sweep"]["evaluations"]
            matching = [e for e in evals
                       if e["K"] == K and e["predictor"] == pred and e["regime"] == "stable"]
            if matching:
                accepted = sum(1 for e in matching if e.get("accepted_strict", False))
                rate = accepted / len(matching)
                rates.append(rate)
        if rates:
            mean_rate = np.mean(rates)
            std_rate = np.std(rates)
            parts.append(f"{mean_rate*100:6.1f}% +/- {std_rate*100:5.1f}%")
        else:
            parts.append(f"{'N/A':>20s}")
    print(f"{K:5d} | {parts[0]:>20s} | {parts[1]:>20s} | {parts[2]:>20s}")

# Momentum catastrophe headline numbers
print("\nMOMENTUM CATASTROPHE HEADLINE:")
for K in K_values:
    predicted_all = []
    actual_all = []
    for seed in seeds:
        evals = data[seed]["depth_sweep"]["evaluations"]
        matching = [e for e in evals if e["K"] == K and e["predictor"] == "momentum"]
        for e in matching:
            predicted_all.append(e["predicted_loss"])
            actual_all.append(e["current_loss"])
    ratio = np.mean(predicted_all) / np.mean(actual_all)
    print(f"  K={K:3d}: predicted/actual ratio = {ratio:.2f}x")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
