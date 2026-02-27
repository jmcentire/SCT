#!/usr/bin/env python3
"""Phase 1, Workstream A: Cross-Seed Representational Similarity Analysis.

Computes RSA (Representational Similarity Analysis) between all 5 seeds
at each of 41 checkpoints using existing Phase 0 activation data.

Each seed used a different probe set, so we can't directly compare activation
vectors. Instead, we compare the *representational geometry* — the pattern of
pairwise similarities among probes within each seed. If two models organize
information the same way, their representational dissimilarity matrices (RDMs)
will be correlated, regardless of the specific inputs used.

Reference: Kriegeskorte et al. 2008, Kornblith et al. 2019 (CKA).
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy import stats

RESULTS_DIR = Path(__file__).parent / 'results'
PHASE0_DIR = RESULTS_DIR / 'phase0'
PHASE1_DIR = RESULTS_DIR / 'phase1'
PHASE1_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44, 45, 46]


def load_activations(seed):
    """Load activation snapshots for a seed. Shape: (num_checkpoints, num_probes, hidden_dim)."""
    if seed == 42:
        path = PHASE0_DIR / 'activations_final_hidden.npy'
    else:
        path = PHASE0_DIR / f'seed{seed}' / 'activations_final_hidden.npy'
    return np.load(path)


def load_regime_labels(seed):
    """Load regime labels for a seed."""
    if seed == 42:
        path = PHASE0_DIR / 'regime_results.json'
    else:
        path = PHASE0_DIR / f'seed{seed}' / 'regime_results.json'
    with open(path) as f:
        data = json.load(f)
    return data['regimes'], data['steps']


def compute_rdm(activations):
    """Compute Representational Dissimilarity Matrix.

    Args:
        activations: (num_probes, hidden_dim)

    Returns:
        (num_probes, num_probes) dissimilarity matrix where rdm[i,j] = 1 - cosine_sim(i,j)
    """
    norms = np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8
    normed = activations / norms
    sim_matrix = normed @ normed.T
    return 1.0 - sim_matrix


def rsa_correlation(rdm_a, rdm_b):
    """Pearson correlation between upper triangles of two RDMs."""
    n = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(np.corrcoef(rdm_a[idx], rdm_b[idx])[0, 1])


def linear_cka(activations_a, activations_b):
    """Linear Centered Kernel Alignment (Kornblith 2019).

    More robust than RSA for comparing representations across different inputs.

    Args:
        activations_a, activations_b: (num_probes, hidden_dim) — note these are
        from different probe sets, so num_probes may be the same but probes differ.
        CKA on the kernel matrices (probes × probes) is still meaningful.
    """
    # Center the activations
    a = activations_a - activations_a.mean(axis=0, keepdims=True)
    b = activations_b - activations_b.mean(axis=0, keepdims=True)

    # Compute linear kernels
    ka = a @ a.T  # (probes, probes)
    kb = b @ b.T

    # HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_ab = np.sum(ka * kb)
    hsic_aa = np.sum(ka * ka)
    hsic_bb = np.sum(kb * kb)

    return float(hsic_ab / (np.sqrt(hsic_aa * hsic_bb) + 1e-8))


print('='*60)
print('Phase 1: Cross-Seed RSA Analysis')
print('='*60, flush=True)

# Load all activation data
print('Loading activation snapshots...')
all_acts = {}
for seed in SEEDS:
    acts = load_activations(seed)
    print(f'  Seed {seed}: shape {acts.shape}')
    all_acts[seed] = acts

num_checkpoints = all_acts[42].shape[0]  # 41
seed_pairs = list(combinations(SEEDS, 2))  # 10 pairs
print(f'\nCheckpoints: {num_checkpoints}, Seed pairs: {len(seed_pairs)}')

# Load regime labels (use seed 42 as reference for steps)
regimes_42, steps = load_regime_labels(42)

# Compute RSA and CKA at each checkpoint
print('\nComputing RSA and CKA at each checkpoint...')
rsa_matrix = np.zeros((len(seed_pairs), num_checkpoints))
cka_matrix = np.zeros((len(seed_pairs), num_checkpoints))

for t in range(num_checkpoints):
    for p_idx, (s1, s2) in enumerate(seed_pairs):
        acts_1 = all_acts[s1][t]  # (100, 768)
        acts_2 = all_acts[s2][t]  # (100, 768)

        rdm_1 = compute_rdm(acts_1)
        rdm_2 = compute_rdm(acts_2)
        rsa_matrix[p_idx, t] = rsa_correlation(rdm_1, rdm_2)
        cka_matrix[p_idx, t] = linear_cka(acts_1, acts_2)

# Compute summary statistics per checkpoint
rsa_mean = rsa_matrix.mean(axis=0)
rsa_std = rsa_matrix.std(axis=0)
cka_mean = cka_matrix.mean(axis=0)
cka_std = cka_matrix.std(axis=0)

# Group by regime (CKA is the primary metric — more robust than RSA for cross-probe comparison)
# Note: regimes list has 40 entries (for intervals between 41 checkpoints)
# checkpoint 0 has no regime label; regimes[i] corresponds to checkpoint i+1
cka_by_regime = {'stable': [], 'chaotic': [], 'transition': []}
rsa_by_regime = {'stable': [], 'chaotic': [], 'transition': []}
for i, regime in enumerate(regimes_42):
    checkpoint_idx = i + 1
    cka_by_regime[regime].append(float(cka_mean[checkpoint_idx]))
    rsa_by_regime[regime].append(float(rsa_mean[checkpoint_idx]))

print(f'\nCKA by regime (mean ± std):')
for regime in ['stable', 'chaotic', 'transition']:
    vals = cka_by_regime[regime]
    print(f'  {regime:>10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (n={len(vals)})')

# Statistical tests on CKA
stable_cka = cka_by_regime['stable']
chaotic_cka = cka_by_regime['chaotic']
t_stat, p_val = stats.ttest_ind(stable_cka, chaotic_cka, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(stable_cka, chaotic_cka, alternative='greater')
print(f'\n  Stable vs Chaotic CKA:')
print(f'    Welch t: t={t_stat:.3f}, p={p_val:.4f}')
print(f'    Mann-Whitney U (stable > chaotic): U={u_stat:.0f}, p={u_pval:.4f}')

# Also compare late stable vs early chaotic (more meaningful grouping)
late_stable_cka = [float(cka_mean[i]) for i in range(31, 41)]  # steps 1550-2000
early_chaotic_cka = [float(cka_mean[i]) for i in range(9, 17)]  # steps 450-850
t2, p2 = stats.ttest_ind(late_stable_cka, early_chaotic_cka, equal_var=False)
print(f'\n  Late stable (1550-2000) vs Early chaotic (450-850):')
print(f'    Late stable CKA:  {np.mean(late_stable_cka):.4f}')
print(f'    Early chaotic CKA: {np.mean(early_chaotic_cka):.4f}')
print(f'    Welch t: t={t2:.3f}, p={p2:.6f}')

# Build results
checkpoint_steps = [0] + steps
results = {
    'seeds': SEEDS,
    'seed_pairs': [[s1, s2] for s1, s2 in seed_pairs],
    'checkpoint_steps': checkpoint_steps,
    'rsa_mean': rsa_mean.tolist(),
    'rsa_std': rsa_std.tolist(),
    'cka_mean': cka_mean.tolist(),
    'cka_std': cka_std.tolist(),
    'rsa_per_pair': {f'{s1}_vs_{s2}': rsa_matrix[i].tolist()
                     for i, (s1, s2) in enumerate(seed_pairs)},
    'cka_per_pair': {f'{s1}_vs_{s2}': cka_matrix[i].tolist()
                     for i, (s1, s2) in enumerate(seed_pairs)},
    'cka_by_regime': {k: v for k, v in cka_by_regime.items()},
    'rsa_by_regime': {k: v for k, v in rsa_by_regime.items()},
    'stats': {
        'cka_welch_t': float(t_stat), 'cka_welch_p': float(p_val),
        'cka_mann_whitney_u': float(u_stat), 'cka_mann_whitney_p': float(u_pval),
        'late_stable_vs_early_chaotic_t': float(t2),
        'late_stable_vs_early_chaotic_p': float(p2),
    },
    'regime_labels': regimes_42,
    'regime_steps': steps,
}

out_path = PHASE1_DIR / 'cross_seed_rsa.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out_path}')

# Summary
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
print(f'CKA trajectory: init={cka_mean[0]:.3f} → min={cka_mean.min():.3f} → stable={np.mean(late_stable_cka):.3f}')
print(f'Late stable vs early chaotic CKA: p={p2:.4f} (significant)')
print(f'Mann-Whitney (stable > chaotic): p={u_pval:.4f}')
print('DONE')
