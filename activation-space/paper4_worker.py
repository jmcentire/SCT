"""Paper 4: Single INLP experiment worker.

Usage:
  python3 -u paper4_worker.py --activations FILE --label NAME --method standard
  python3 -u paper4_worker.py --activations FILE --label NAME --method shaped --noise-scale 0.1
  python3 -u paper4_worker.py --activations FILE --label NAME --method isotropic --noise-scale 0.1
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

# 160 probes: 4 shapes x 4 domains x 10
# Labels are deterministic from probe order
def make_labels():
    shape_labels = []
    domain_labels = []
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


def run_inlp(X, domain_labels, shape_labels, label, max_iter=50, threshold=0.30,
             noise_fn=None):
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

        if noise_fn is not None:
            X_train, y_train = noise_fn(X, domain_labels, removed)
        else:
            X_train, y_train = X, domain_labels

        clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        clf.fit(X_train, y_train)
        W = clf.coef_
        _, _, Vt = np.linalg.svd(W, full_matrices=False)
        d = Vt[0]; d = d / np.linalg.norm(d)
        X = X - np.outer(X @ d, d)
        removed.append(d)
    return np.array(removed), history


def make_shaped_noise_fn(shape_labels, noise_scale, n_samples=5):
    def fn(X, domain_labels, removed):
        shape_clf = LogisticRegression(max_iter=5000, solver='lbfgs')
        shape_clf.fit(X, shape_labels)
        W_shape = shape_clf.coef_
        _, _, Vt = np.linalg.svd(W_shape, full_matrices=False)
        S_basis = Vt[:len(SHAPES)-1]
        P_S = S_basis.T @ S_basis
        P_S_perp = np.eye(X.shape[1]) - P_S
        act_std = np.std(X)
        X_all = [X.copy()]
        for _ in range(n_samples):
            noise = np.random.randn(*X.shape) * act_std * noise_scale
            X_all.append(X + noise @ P_S_perp)
        return np.vstack(X_all), np.tile(domain_labels, n_samples + 1)
    return fn


def make_isotropic_noise_fn(noise_scale, n_samples=5):
    def fn(X, domain_labels, removed):
        act_std = np.std(X)
        X_all = [X.copy()]
        for _ in range(n_samples):
            noise = np.random.randn(*X.shape) * act_std * noise_scale
            X_all.append(X + noise)
        return np.vstack(X_all), np.tile(domain_labels, n_samples + 1)
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activations', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--method', required=True, choices=['standard', 'shaped', 'isotropic'])
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--out-dir', default='results/paper4_shaped')
    args = parser.parse_args()

    shape_labels, domain_labels = make_labels()
    act = np.load(args.activations)
    np.random.seed(42)

    print(f'[{args.label}] Loaded activations: {act.shape}', flush=True)
    print(f'[{args.label}] Method: {args.method}, noise_scale: {args.noise_scale}', flush=True)

    if args.method == 'shaped':
        noise_fn = make_shaped_noise_fn(shape_labels, args.noise_scale)
    elif args.method == 'isotropic':
        noise_fn = make_isotropic_noise_fn(args.noise_scale)
    else:
        noise_fn = None

    t0 = time.time()
    dirs, history = run_inlp(act, domain_labels, shape_labels, args.label, noise_fn=noise_fn)
    elapsed = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f'{args.label}_dirs.npy', dirs)
    with open(out_dir / f'{args.label}_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    last = history[-1]
    first_s = history[0]['shape']
    survival = last['shape'] / first_s if first_s > 0 else 0

    print(f'\n[{args.label}] COMPLETE: {len(dirs)} dirs, domain={last["domain"]:.3f}, shape={last["shape"]:.3f}, survival={survival:.1%}, {elapsed:.0f}s', flush=True)


if __name__ == '__main__':
    main()
