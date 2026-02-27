#!/usr/bin/env python3
"""Run Phase 0 with multiple seeds using subprocess + retry for crash resilience."""
import os, sys, json, subprocess, time
import numpy as np

os.chdir(os.path.dirname(__file__))

SEEDS = [43, 44, 45, 46]
MAX_RETRIES = 5

print('='*60)
print('Phase 0: Multi-Seed Reproducibility Run')
print(f'Seeds: {SEEDS}, max retries per seed: {MAX_RETRIES}')
print('='*60, flush=True)

env = os.environ.copy()
env['TOKENIZERS_PARALLELISM'] = 'false'
env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env['HF_HUB_OFFLINE'] = '1'

for seed in SEEDS:
    print(f'\n--- SEED {seed} ---', flush=True)
    for attempt in range(1, MAX_RETRIES + 1):
        print(f'  Attempt {attempt}/{MAX_RETRIES}...', flush=True)
        result = subprocess.run(
            [sys.executable, '-u', 'run_single_seed.py', str(seed)],
            env=env, timeout=3600,
            capture_output=False,
        )
        if result.returncode == 0:
            print(f'  Seed {seed} SUCCESS', flush=True)
            break
        else:
            print(f'  Seed {seed} crashed (exit {result.returncode}), retrying...', flush=True)
            time.sleep(2)
    else:
        print(f'  Seed {seed} FAILED after {MAX_RETRIES} attempts', flush=True)

# Cross-seed comparison
print(f'\n{"="*60}')
print('CROSS-SEED COMPARISON')
print(f'{"="*60}', flush=True)

all_seed_results = {}

# Seed 42 (original)
try:
    r42 = json.load(open('results/phase0/regime_results.json'))
    m42 = json.load(open('results/phase0/training_meta.json'))
    all_seed_results[42] = {
        'stable_frac': r42['stable_fraction'],
        'chaotic_frac': r42['chaotic_fraction'],
        'similarities': r42['similarities'],
        'final_loss': m42['losses'][-1],
    }
except Exception:
    pass

for seed in SEEDS:
    try:
        r = json.load(open(f'results/phase0/seed{seed}/regime_results.json'))
        m = json.load(open(f'results/phase0/seed{seed}/training_meta.json'))
        all_seed_results[seed] = {
            'stable_frac': r['stable_fraction'],
            'chaotic_frac': r['chaotic_fraction'],
            'similarities': r['similarities'],
            'final_loss': m['losses'][-1],
        }
    except Exception as e:
        print(f'  Seed {seed}: no results ({e})')

if len(all_seed_results) < 2:
    print('Not enough seeds completed for comparison.')
    sys.exit(1)

print(f'\n{"Seed":>6} {"Stable":>8} {"Chaotic":>8} {"Final Loss":>12} {"Mean Sim":>10}')
print('-' * 50)
for s in sorted(all_seed_results.keys()):
    r = all_seed_results[s]
    mean_sim = np.mean(r['similarities'])
    print(f'{s:>6} {r["stable_frac"]:>7.1%} {r["chaotic_frac"]:>7.1%} '
          f'{r["final_loss"]:>11.4f} {mean_sim:>9.4f}')

# Correlation
all_sims = []
labels = sorted(all_seed_results.keys())
for s in labels:
    all_sims.append(all_seed_results[s]['similarities'])

min_len = min(len(s) for s in all_sims)
sim_matrix = np.array([s[:min_len] for s in all_sims])

print(f'\nSimilarity trajectory correlations:')
corrs = []
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        corr = np.corrcoef(sim_matrix[i], sim_matrix[j])[0, 1]
        corrs.append(corr)
        print(f'  Seed {labels[i]} vs {labels[j]}: r={corr:.3f}')

mean_corr = np.mean(corrs) if corrs else 0
print(f'\n  Mean pairwise correlation: r={mean_corr:.3f}')

summary = {
    'seeds': [int(s) for s in labels],
    'per_seed': {str(k): {kk: vv for kk, vv in v.items() if kk != 'similarities'}
                 for k, v in all_seed_results.items()},
    'mean_pairwise_correlation': float(mean_corr),
}
with open('results/phase0/cross_seed_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSaved to results/phase0/cross_seed_summary.json')
print('DONE')
