"""Single SR experiment: perturb activations ONCE, then clean INLP.

Usage: python3 -u paper4_sr_worker.py --activations FILE --label NAME --noise-type shaped --sigma 0.01
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import json, time, argparse
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
    return cross_val_score(clf, X, labels, cv=cv, scoring='accuracy').mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--noise-type', required=True, choices=['shaped', 'isotropic', 'none'])
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--out-dir', default='results/paper4_shaped')
    args = parser.parse_args()

    np.random.seed(42)
    shape_labels, domain_labels = make_labels()
    act = np.load(args.activations)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    hist_file = out_dir / f'{args.label}_history.json'
    if hist_file.exists():
        print(f'[{args.label}] Already done, skipping.', flush=True)
        return

    print(f'[{args.label}] Loaded: {act.shape}, noise={args.noise_type}, sigma={args.sigma}', flush=True)

    # Perturb activations ONCE
    if args.noise_type == 'none':
        act_pert = act.copy()
    else:
        act_std = np.std(act)
        noise = np.random.randn(*act.shape) * act_std * args.sigma
        if args.noise_type == 'shaped':
            # Project noise orthogonal to shape subspace
            shape_clf = LogisticRegression(max_iter=5000, solver='lbfgs')
            shape_clf.fit(act, shape_labels)
            _, _, Vt = np.linalg.svd(shape_clf.coef_, full_matrices=False)
            S_basis = Vt[:len(SHAPES)-1]
            P_S_perp = np.eye(act.shape[1]) - S_basis.T @ S_basis
            noise = noise @ P_S_perp
        act_pert = act + noise

    d0 = classify(act_pert, domain_labels)
    s0 = classify(act_pert, shape_labels)
    print(f'[{args.label}] Post-perturbation: domain={d0:.3f}, shape={s0:.3f}', flush=True)

    # Standard INLP on perturbed data
    X = act_pert.copy()
    removed, history = [], []
    t0 = time.time()
    for it in range(50):
        d_acc = classify(X, domain_labels)
        s_acc = classify(X, shape_labels)
        history.append({'iter': it, 'domain': float(d_acc), 'shape': float(s_acc)})
        print(f'  [{args.label}] INLP {it:2d}: domain={d_acc:.3f}, shape={s_acc:.3f}', flush=True)
        if d_acc < 0.30:
            break
        clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        clf.fit(X, domain_labels)
        _, _, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
        d = Vt[0]; d /= np.linalg.norm(d)
        X -= np.outer(X @ d, d)
        removed.append(d)
    elapsed = time.time() - t0

    with open(hist_file, 'w') as f:
        json.dump(history, f, indent=2)
    np.save(out_dir / f'{args.label}_dirs.npy', np.array(removed))

    last = history[-1]
    surv = last['shape'] / history[0]['shape'] if history[0]['shape'] > 0 else 0
    print(f'\n[{args.label}] COMPLETE: {len(removed)} dirs, domain={last["domain"]:.3f}, shape={last["shape"]:.3f}, survival={surv:.1%}, {elapsed:.0f}s', flush=True)

if __name__ == '__main__':
    main()
