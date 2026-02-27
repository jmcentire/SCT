#!/usr/bin/env python3
"""Phase 2c: Multi-seed wrapper + aggregation.

Runs phase2c_depth_sweep.py for seeds 42-46 sequentially,
then aggregates into results/phase2c/aggregate.json.

Usage:
  python run_phase2c.py                    # Run all 5 seeds + aggregate
  python run_phase2c.py --seeds 42 43      # Run specific seeds
  python run_phase2c.py --aggregate-only   # Just aggregate existing results
  python run_phase2c.py --max-steps 200    # Smoke test
"""

import os, sys, json, subprocess, time, argparse
import numpy as np
from pathlib import Path


SEEDS = [42, 43, 44, 45, 46]
K_SWEEP_VALUES = [5, 10, 25, 50, 75, 100]
PREDICTORS = ['momentum', 'linear', 'quadratic']
SWEEP_REGIMES = ['stable', 'transition']
CASCADE_CONFIGS = [(4, 25), (2, 50), (10, 10)]


def run_seed(seed, max_steps=2000, extra_args=None):
    """Run a single seed as a subprocess."""
    cmd = [sys.executable, 'phase2c_depth_sweep.py', '--seed', str(seed),
           '--max-steps', str(max_steps)]
    if extra_args:
        cmd.extend(extra_args)

    print(f'\n{"=" * 60}')
    print(f'Running seed {seed}...')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"=" * 60}', flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.')
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'WARNING: Seed {seed} exited with code {result.returncode}', flush=True)
        return False, elapsed
    else:
        print(f'Seed {seed} completed in {elapsed:.0f}s', flush=True)
        return True, elapsed


def aggregate_results(seeds, base_dir=None):
    """Aggregate per-seed results into aggregate.json."""
    if base_dir is None:
        base_dir = Path('results/phase2c')

    print(f'\n{"=" * 60}')
    print('Aggregating results...')
    print(f'{"=" * 60}', flush=True)

    all_results = {}
    for seed in seeds:
        results_path = base_dir / f'seed{seed}' / 'combined_results.json'
        if results_path.exists():
            with open(results_path) as f:
                all_results[seed] = json.load(f)
            print(f'  Loaded seed {seed}')
        else:
            print(f'  WARNING: No results for seed {seed} at {results_path}')

    if not all_results:
        print('No results to aggregate!')
        return None

    completed_seeds = sorted(all_results.keys())
    print(f'Aggregating {len(completed_seeds)} seeds: {completed_seeds}', flush=True)

    # ---- Depth curve: acceptance rate vs K, per predictor, per regime, with error bars ----
    depth_curve = {}
    for predictor in PREDICTORS:
        depth_curve[predictor] = {}
        for regime in SWEEP_REGIMES:
            per_seed_rates_strict = {K: [] for K in K_SWEEP_VALUES}
            per_seed_rates_adaptive = {K: [] for K in K_SWEEP_VALUES}
            per_seed_rates_pct = {K: [] for K in K_SWEEP_VALUES}

            for seed, data in all_results.items():
                summary = data.get('depth_sweep', {}).get('summary_by_k', {})
                for K in K_SWEEP_VALUES:
                    k_data = summary.get(str(K), summary.get(K, {}))
                    pred_data = k_data.get(predictor, {})
                    regime_data = pred_data.get(regime, {})
                    if regime_data:
                        per_seed_rates_strict[K].append(regime_data.get('strict_rate', 0))
                        per_seed_rates_adaptive[K].append(regime_data.get('adaptive_rate', 0))
                        per_seed_rates_pct[K].append(regime_data.get('pct_rate', 0))

            depth_curve[predictor][regime] = {
                'k_values': K_SWEEP_VALUES,
                'strict': {
                    'mean': [float(np.mean(per_seed_rates_strict[K])) if per_seed_rates_strict[K] else 0
                             for K in K_SWEEP_VALUES],
                    'std': [float(np.std(per_seed_rates_strict[K])) if per_seed_rates_strict[K] else 0
                            for K in K_SWEEP_VALUES],
                    'per_seed': {str(K): per_seed_rates_strict[K] for K in K_SWEEP_VALUES},
                },
                'adaptive': {
                    'mean': [float(np.mean(per_seed_rates_adaptive[K])) if per_seed_rates_adaptive[K] else 0
                             for K in K_SWEEP_VALUES],
                    'std': [float(np.std(per_seed_rates_adaptive[K])) if per_seed_rates_adaptive[K] else 0
                            for K in K_SWEEP_VALUES],
                },
                'pct': {
                    'mean': [float(np.mean(per_seed_rates_pct[K])) if per_seed_rates_pct[K] else 0
                             for K in K_SWEEP_VALUES],
                    'std': [float(np.std(per_seed_rates_pct[K])) if per_seed_rates_pct[K] else 0
                            for K in K_SWEEP_VALUES],
                },
            }

    # ---- Cascade summary ----
    cascade_summary = {}
    for chain_length, K_per_link in CASCADE_CONFIGS:
        key = f'{chain_length}x{K_per_link}'
        all_drifts = []  # List of per-link drift lists across seeds
        all_final_losses = []

        for seed, data in all_results.items():
            cascades = data.get('cascade_results', {}).get('configurations', [])
            for c in cascades:
                if c['chain_length'] == chain_length and c['K_per_link'] == K_per_link:
                    if c['links']:
                        drifts = [l['l2_drift'] for l in c['links']]
                        all_drifts.append(drifts)
                        all_final_losses.append(c['links'][-1]['trained_loss'])

        if all_drifts:
            # Pad to same length
            max_links = max(len(d) for d in all_drifts)
            padded = []
            for d in all_drifts:
                padded.append(d + [None] * (max_links - len(d)))

            mean_drift_per_link = []
            std_drift_per_link = []
            for link_idx in range(max_links):
                vals = [p[link_idx] for p in padded if p[link_idx] is not None]
                mean_drift_per_link.append(float(np.mean(vals)) if vals else 0)
                std_drift_per_link.append(float(np.std(vals)) if vals else 0)

            cascade_summary[key] = {
                'n_runs': len(all_drifts),
                'mean_l2_per_link': mean_drift_per_link,
                'std_l2_per_link': std_drift_per_link,
                'mean_final_loss': float(np.mean(all_final_losses)) if all_final_losses else 0,
                'std_final_loss': float(np.std(all_final_losses)) if all_final_losses else 0,
            }

    # ---- Acceptance comparison: strict vs adaptive vs pct ----
    acceptance_comparison = {}
    for criterion in ['strict', 'adaptive', 'pct']:
        by_k_mean = []
        by_k_std = []
        for K in K_SWEEP_VALUES:
            seed_rates = []
            for seed, data in all_results.items():
                evals = data.get('depth_sweep', {}).get('evaluations', [])
                k_evals = [e for e in evals if e['K'] == K and e['predictor'] == 'momentum'
                           and e['regime'] == 'stable']
                if k_evals:
                    key = f'accepted_{criterion}'
                    rate = sum(1 for e in k_evals if e.get(key, False)) / len(k_evals)
                    seed_rates.append(rate)
            by_k_mean.append(float(np.mean(seed_rates)) if seed_rates else 0)
            by_k_std.append(float(np.std(seed_rates)) if seed_rates else 0)
        acceptance_comparison[criterion] = {
            'k_values': K_SWEEP_VALUES,
            'mean': by_k_mean,
            'std': by_k_std,
        }

    # ---- Summary headline numbers ----
    # Max safe K: largest K with >50% strict acceptance (momentum, stable)
    max_safe_K = 0
    mom_stable = depth_curve.get('momentum', {}).get('stable', {}).get('strict', {})
    if mom_stable.get('mean'):
        for i, K in enumerate(K_SWEEP_VALUES):
            if mom_stable['mean'][i] > 0.5:
                max_safe_K = K

    # Mean hypothetical speedup
    speedups = []
    for seed, data in all_results.items():
        trained = data.get('summary', {}).get('total_steps_trained', 0)
        skipped = data.get('summary', {}).get('total_steps_skipped', 0)
        total = trained + skipped
        if total > 0:
            speedups.append(skipped / total)

    # Best predictor across seeds
    best_predictor = 'momentum'
    best_mean_rate = 0
    for pred in PREDICTORS:
        rates = depth_curve.get(pred, {}).get('stable', {}).get('strict', {}).get('mean', [])
        if rates:
            avg = np.mean(rates[:3])  # Average over K=5,10,25
            if avg > best_mean_rate:
                best_mean_rate = avg
                best_predictor = pred

    summary = {
        'seeds': completed_seeds,
        'n_seeds': len(completed_seeds),
        'max_safe_K': max_safe_K,
        'best_predictor': best_predictor,
        'mean_hypothetical_speedup': float(np.mean(speedups)) if speedups else 0,
        'std_hypothetical_speedup': float(np.std(speedups)) if speedups else 0,
    }

    aggregate = {
        'seeds': completed_seeds,
        'depth_curve': {'by_predictor': depth_curve},
        'cascade_summary': cascade_summary,
        'acceptance_comparison': acceptance_comparison,
        'summary': summary,
    }

    agg_path = base_dir / 'aggregate.json'
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f'\nAggregate saved to {agg_path}', flush=True)

    # Print headline numbers
    print(f'\n{"=" * 60}')
    print('AGGREGATE SUMMARY')
    print(f'{"=" * 60}')
    print(f'Seeds completed: {completed_seeds}')
    print(f'Best predictor: {best_predictor}')
    print(f'Max safe K (>50% acceptance): {max_safe_K}')
    print(f'Mean hypothetical speedup: {summary["mean_hypothetical_speedup"]:.1%} '
          f'+/- {summary["std_hypothetical_speedup"]:.1%}')

    print(f'\nDepth curve (momentum, stable, strict):')
    ms = depth_curve.get('momentum', {}).get('stable', {}).get('strict', {})
    for i, K in enumerate(K_SWEEP_VALUES):
        m = ms.get('mean', [0]*len(K_SWEEP_VALUES))[i]
        s = ms.get('std', [0]*len(K_SWEEP_VALUES))[i]
        print(f'  K={K:>3}: {m:.1%} +/- {s:.1%}')

    # Generate aggregate plots
    print('\nGenerating aggregate plots...')
    try:
        from src.analysis.plot_phase2c import plot_aggregate
        import matplotlib
        matplotlib.use('Agg')
        plot_aggregate(aggregate, save_dir=str(base_dir))
        print('Aggregate plots saved.')
    except Exception as e:
        print(f'Plot generation failed: {e}')
        import traceback
        traceback.print_exc()

    return aggregate


def main():
    parser = argparse.ArgumentParser(description='Phase 2c: Multi-seed wrapper')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results(args.seeds)
        return

    extra_args = []
    if args.cleanup:
        extra_args.append('--cleanup')

    results = {}
    for seed in args.seeds:
        success, elapsed = run_seed(seed, max_steps=args.max_steps, extra_args=extra_args)
        results[seed] = {'success': success, 'time': elapsed}

    print(f'\n{"=" * 60}')
    print('SEED SUMMARY')
    print(f'{"=" * 60}')
    for seed, r in results.items():
        status = 'OK' if r['success'] else 'FAIL'
        print(f'  Seed {seed}: {status} ({r["time"]:.0f}s)')

    aggregate_results(args.seeds)
    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
