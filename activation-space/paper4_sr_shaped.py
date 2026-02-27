"""Paper 4: Shaped SR — perturb geometry ONCE, then run clean INLP.

This is the correct analog to Paper 3's stochastic resonance:
- Paper 3 SR: add noise to centroids before Gram-Schmidt
- Here: add shaped noise to activations before INLP

The noise is shaped: projected orthogonal to the structural subspace,
so it perturbs domain geometry without disturbing shape structure.

Usage: python3 -u paper4_sr_shaped.py --activations FILE
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

SHAPES = ['hierarchical', 'causal', 'constraint', 'evidence']
DOMAINS = ['medical', 'legal', 'code', 'science']


def make_labels():
    shape_labels, domain_labels = [], []
    for shape in SHAPES:
        for domain in DOMAINS:
            for _ in range(10):
                shape_labels.append(shape)
                domain_labels.append(domain)
    return np.array(shape_labels), np.array(domain_labels)


def classify(X, labels, cv=5):
    clf = LogisticRegression(max_iter=5000, solver='lbfgs')
    scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


def standard_inlp(X, domain_labels, shape_labels, label, max_iter=50, threshold=0.30):
    X = X.copy()
    removed = []
    history = []
    for it in range(max_iter):
        d_acc, _ = classify(X, domain_labels)
        s_acc, _ = classify(X, shape_labels)
        history.append({'iter': it, 'domain': float(d_acc), 'shape': float(s_acc)})
        print(f'  [{label}] INLP {it:2d}: domain={d_acc:.3f}, shape={s_acc:.3f}', flush=True)
        if d_acc < threshold:
            break
        clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        clf.fit(X, domain_labels)
        W = clf.coef_
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
        d = Vt[0]; d = d / np.linalg.norm(d)
        X = X - np.outer(X @ d, d)
        removed.append(d)
    return np.array(removed), history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True)
    parser.add_argument('--out-dir', default='results/paper4_shaped')
    args = parser.parse_args()

    np.random.seed(42)
    shape_labels, domain_labels = make_labels()
    act = np.load(args.activations)
    print(f'Loaded activations: {act.shape}', flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compute shape subspace for noise shaping
    shape_clf = LogisticRegression(max_iter=5000, solver='lbfgs')
    shape_clf.fit(act, shape_labels)
    W_shape = shape_clf.coef_
    _, _, Vt_shape = np.linalg.svd(W_shape, full_matrices=False)
    S_basis = Vt_shape[:len(SHAPES)-1]  # k-1 shape directions
    P_S = S_basis.T @ S_basis
    P_S_perp = np.eye(act.shape[1]) - P_S
    print(f'Shape subspace: {S_basis.shape[0]} directions', flush=True)

    # Baseline shape/domain accuracy
    d0, _ = classify(act, domain_labels)
    s0, _ = classify(act, shape_labels)
    print(f'Baseline: domain={d0:.3f}, shape={s0:.3f}', flush=True)

    noise_scales = [0.005, 0.01, 0.02, 0.05, 0.1]
    act_std = np.std(act)

    for sigma in noise_scales:
        label = f'SR_shaped_{sigma}'
        print(f'\n{"="*60}')
        print(f'{label}: sigma={sigma} (absolute={sigma*act_std:.4f})')
        print(f'{"="*60}', flush=True)

        # Add shaped noise ONCE to the activations
        noise = np.random.randn(*act.shape) * act_std * sigma
        noise_shaped = noise @ P_S_perp  # project out shape directions
        act_perturbed = act + noise_shaped

        # Check that shape is preserved after perturbation
        d_pert, _ = classify(act_perturbed, domain_labels)
        s_pert, _ = classify(act_perturbed, shape_labels)
        print(f'After perturbation: domain={d_pert:.3f}, shape={s_pert:.3f}', flush=True)

        # Run clean standard INLP on perturbed activations
        t0 = time.time()
        dirs, history = standard_inlp(act_perturbed, domain_labels, shape_labels, label)
        elapsed = time.time() - t0

        with open(out_dir / f'{label}_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        np.save(out_dir / f'{label}_dirs.npy', dirs)

        last = history[-1]
        surv = last['shape'] / history[0]['shape'] if history[0]['shape'] > 0 else 0
        print(f'\n[{label}] COMPLETE: {len(dirs)} dirs, domain={last["domain"]:.3f}, shape={last["shape"]:.3f}, survival={surv:.1%}, {elapsed:.0f}s', flush=True)

    # Also run with isotropic noise for comparison
    for sigma in [0.01, 0.02, 0.05]:
        label = f'SR_isotropic_{sigma}'
        print(f'\n{"="*60}')
        print(f'{label}: sigma={sigma}')
        print(f'{"="*60}', flush=True)

        noise = np.random.randn(*act.shape) * act_std * sigma
        act_perturbed = act + noise  # no shaping

        d_pert, _ = classify(act_perturbed, domain_labels)
        s_pert, _ = classify(act_perturbed, shape_labels)
        print(f'After perturbation: domain={d_pert:.3f}, shape={s_pert:.3f}', flush=True)

        t0 = time.time()
        dirs, history = standard_inlp(act_perturbed, domain_labels, shape_labels, label)
        elapsed = time.time() - t0

        with open(out_dir / f'{label}_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        last = history[-1]
        surv = last['shape'] / history[0]['shape'] if history[0]['shape'] > 0 else 0
        print(f'\n[{label}] COMPLETE: {len(dirs)} dirs, domain={last["domain"]:.3f}, shape={last["shape"]:.3f}, survival={surv:.1%}, {elapsed:.0f}s', flush=True)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60, flush=True)
    print(f'{"Method":<25} {"Iters":>5} {"Final Domain":>13} {"Final Shape":>12} {"Survival":>10}')
    for f in sorted(out_dir.glob('SR_*_history.json')):
        name = f.stem.replace('_history', '')
        h = json.load(open(f))
        last = h[-1]
        surv = last['shape'] / h[0]['shape'] if h[0]['shape'] > 0 else 0
        print(f'{name:<25} {last["iter"]:>5d} {last["domain"]:>13.3f} {last["shape"]:>12.3f} {surv:>9.1%}')


if __name__ == '__main__':
    main()
