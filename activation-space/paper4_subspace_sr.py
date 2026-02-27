"""Paper 4: Subspace-targeted SR.

The right experiment: decompose activations into domain and structural components
using the already-computed INLP directions, apply SR noise within the domain
subspace only (shaped to its variance structure), reconstruct, then run clean INLP.

Control: same but noise in structural subspace, domain untouched.

Usage: python3 -u paper4_subspace_sr.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json, time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SHAPES = ['hierarchical', 'causal', 'constraint', 'evidence']
DOMAINS = ['medical', 'legal', 'code', 'science']

def make_labels():
    s, d = [], []
    for shape in SHAPES:
        for domain in DOMAINS:
            for _ in range(10):
                s.append(shape); d.append(domain)
    return np.array(s), np.array(d)

def classify(X, labels, cv=5):
    clf = LogisticRegression(max_iter=5000, solver='lbfgs')
    return cross_val_score(clf, X, labels, cv=cv, scoring='accuracy', n_jobs=5).mean()

def standard_inlp(X, domain_labels, shape_labels, label, max_iter=50, threshold=0.30):
    X = X.copy()
    removed, history = [], []
    for it in range(max_iter):
        d_acc = classify(X, domain_labels)
        s_acc = classify(X, shape_labels)
        history.append({'iter': it, 'domain': float(d_acc), 'shape': float(s_acc)})
        print(f'  [{label}] INLP {it:2d}: domain={d_acc:.3f}, shape={s_acc:.3f}', flush=True)
        if d_acc < threshold:
            break
        clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        clf.fit(X, domain_labels)
        _, _, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
        d = Vt[0]; d /= np.linalg.norm(d)
        X -= np.outer(X @ d, d)
        removed.append(d)
    return np.array(removed), history


def main():
    np.random.seed(42)
    shape_labels, domain_labels = make_labels()

    out_dir = Path('results/paper4_shaped')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 7B activations and pre-computed INLP directions
    act = np.load(out_dir / 'Qwen2.5-7B_activations.npy')
    inlp_dirs = np.load(out_dir / 'Qwen2.5-7B_standard_dirs.npy')
    print(f'Activations: {act.shape}', flush=True)
    print(f'INLP directions: {inlp_dirs.shape}', flush=True)

    # Build domain subspace projector from INLP directions
    # These are already orthogonal by construction (each found in null space of previous)
    D_basis = inlp_dirs  # (n_dirs, 3584) — each row is a unit direction
    P_D = D_basis.T @ D_basis  # projection onto domain subspace
    P_D_perp = np.eye(act.shape[1]) - P_D  # complement (structural + everything else)

    # Decompose each activation
    act_domain = act @ P_D      # domain component
    act_structural = act @ P_D_perp  # structural component (+ residual)

    print(f'Domain component norm: {np.linalg.norm(act_domain, axis=1).mean():.2f}', flush=True)
    print(f'Structural component norm: {np.linalg.norm(act_structural, axis=1).mean():.2f}', flush=True)
    print(f'Reconstruction error: {np.linalg.norm(act - (act_domain + act_structural)):.6f}', flush=True)

    # Compute variance structure of domain subspace
    # Project activations into the domain subspace coordinates
    act_domain_coords = act @ D_basis.T  # (160, n_dirs) — coordinates in domain subspace
    domain_var = np.var(act_domain_coords, axis=0)  # variance along each domain direction
    print(f'Domain subspace variances: min={domain_var.min():.4f}, max={domain_var.max():.4f}, '
          f'mean={domain_var.mean():.4f}', flush=True)

    # Baseline
    d0 = classify(act, domain_labels)
    s0 = classify(act, shape_labels)
    print(f'\nBaseline: domain={d0:.3f}, shape={s0:.3f}', flush=True)

    # ============================================================
    # Experiment 1: SR noise in DOMAIN subspace, structural untouched
    # ============================================================
    sigmas = [0.5, 1.0, 0.2, 0.1, 2.0]
    domain_std = np.std(act_domain_coords)

    for sigma in sigmas:
        label = f'SR_domain_{sigma}'
        hist_file = out_dir / f'{label}_history.json'
        if hist_file.exists():
            print(f'\n[{label}] cached, skipping', flush=True)
            continue

        print(f'\n{"="*60}')
        print(f'{label}: noise in domain subspace, sigma={sigma}*domain_std')
        print(f'{"="*60}', flush=True)

        # Generate noise in domain subspace coordinates, shaped to variance structure
        noise_coords = np.random.randn(act.shape[0], len(inlp_dirs))
        # Scale each direction by its variance (variance-shaped noise)
        noise_coords *= np.sqrt(domain_var + 1e-10) * sigma
        # Map back to full space
        noise_full = noise_coords @ D_basis  # (160, 3584)

        # Reconstruct: perturbed domain + unchanged structural
        act_perturbed = act_structural + (act_domain + noise_full)

        d_pert = classify(act_perturbed, domain_labels)
        s_pert = classify(act_perturbed, shape_labels)
        print(f'Post-perturbation: domain={d_pert:.3f}, shape={s_pert:.3f}', flush=True)

        t0 = time.time()
        dirs, history = standard_inlp(act_perturbed, domain_labels, shape_labels, label)
        elapsed = time.time() - t0

        with open(hist_file, 'w') as f:
            json.dump(history, f, indent=2)

        last = history[-1]
        surv = last['shape'] / history[0]['shape'] if history[0]['shape'] > 0 else 0
        print(f'\n[{label}] COMPLETE: {len(dirs)} dirs, domain={last["domain"]:.3f}, '
              f'shape={last["shape"]:.3f}, survival={surv:.1%}, {elapsed:.0f}s', flush=True)

    # ============================================================
    # Experiment 2: SR noise in STRUCTURAL subspace, domain untouched
    # ============================================================
    # First compute structural subspace from shape classifier
    shape_clf = LogisticRegression(max_iter=5000, solver='lbfgs')
    shape_clf.fit(act, shape_labels)
    _, _, Vt_shape = np.linalg.svd(shape_clf.coef_, full_matrices=False)
    S_basis = Vt_shape[:len(SHAPES)-1]  # (3, 3584)
    print(f'\nStructural basis: {S_basis.shape}', flush=True)

    act_struct_coords = act @ S_basis.T
    struct_var = np.var(act_struct_coords, axis=0)
    struct_std = np.std(act_struct_coords)
    print(f'Structural subspace variances: {struct_var}', flush=True)

    for sigma in [1.0, 0.5, 0.1, 2.0]:
        label = f'SR_structural_{sigma}'
        hist_file = out_dir / f'{label}_history.json'
        if hist_file.exists():
            print(f'\n[{label}] cached, skipping', flush=True)
            continue

        print(f'\n{"="*60}')
        print(f'{label}: noise in STRUCTURAL subspace, domain untouched')
        print(f'{"="*60}', flush=True)

        P_S = S_basis.T @ S_basis
        noise_coords = np.random.randn(act.shape[0], len(S_basis))
        noise_coords *= np.sqrt(struct_var + 1e-10) * sigma
        noise_full = noise_coords @ S_basis

        act_perturbed = act + noise_full  # only structural component perturbed

        d_pert = classify(act_perturbed, domain_labels)
        s_pert = classify(act_perturbed, shape_labels)
        print(f'Post-perturbation: domain={d_pert:.3f}, shape={s_pert:.3f}', flush=True)

        t0 = time.time()
        dirs, history = standard_inlp(act_perturbed, domain_labels, shape_labels, label)
        elapsed = time.time() - t0

        with open(hist_file, 'w') as f:
            json.dump(history, f, indent=2)

        last = history[-1]
        surv = last['shape'] / history[0]['shape'] if history[0]['shape'] > 0 else 0
        print(f'\n[{label}] COMPLETE: {len(dirs)} dirs, domain={last["domain"]:.3f}, '
              f'shape={last["shape"]:.3f}, survival={surv:.1%}, {elapsed:.0f}s', flush=True)

    # ============================================================
    # Summary
    # ============================================================
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}', flush=True)
    print(f'{"Method":<25} {"Iters":>5} {"Final Domain":>13} {"Final Shape":>12} {"Survival":>10}')

    # Include standard for comparison
    std_hist = json.load(open(out_dir / 'Qwen2.5-7B_standard_history.json'))
    last = std_hist[-1]
    surv = last['shape'] / std_hist[0]['shape'] if std_hist[0]['shape'] > 0 else 0
    print(f'{"Standard (baseline)":<25} {last["iter"]:>5d} {last["domain"]:>13.3f} {last["shape"]:>12.3f} {surv:>9.1%}')

    for f in sorted(out_dir.glob('SR_*_history.json')):
        name = f.stem.replace('_history', '')
        h = json.load(open(f))
        last = h[-1]
        surv = last['shape'] / h[0]['shape'] if h[0]['shape'] > 0 else 0
        print(f'{name:<25} {last["iter"]:>5d} {last["domain"]:>13.3f} {last["shape"]:>12.3f} {surv:>9.1%}')


if __name__ == '__main__':
    main()
