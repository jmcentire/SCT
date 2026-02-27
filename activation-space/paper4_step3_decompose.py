"""Paper 4 Step 3: Improved decomposition with INLP + contrastive analysis.

Uses saved shape-labeled activations from step2. Three approaches:
  A) Reduced-rank CCA (PCA to fewer dims than samples)
  B) INLP: iteratively null-project domain classifiers, test shape survival
  C) Centered Kernel Alignment: measure domain vs shape signal strength

Usage: KMP_DUPLICATE_LIB_OK=TRUE python3 paper4_step3_decompose.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import CCA

from paper4_probes import SHAPES, DOMAINS


def load_data():
    """Load saved activations and labels."""
    out_dir = Path('results/paper4')
    activations = np.load(out_dir / 'shape_activations.npy')
    with open(out_dir / 'shape_labels.json') as f:
        data = json.load(f)
    probes = data['probes']
    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])
    return activations, shape_labels, domain_labels


def classify(X, labels, label_name="", cv=5):
    """Run classifier, return mean accuracy."""
    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    scores = cross_val_score(clf, X, labels, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()


# ============================================================
# Approach A: Reduced-rank CCA
# ============================================================

def approach_a_cca(activations, shape_labels, domain_labels):
    """CCA with PCA reduced to fewer dims than samples per pair."""
    print('\n' + '=' * 60)
    print('APPROACH A: REDUCED-RANK CCA')
    print('=' * 60)

    # With 40 samples per domain pair, reduce to 20 PCA dims
    n_pca = 20
    pca = PCA(n_components=n_pca)
    X_pca = pca.fit_transform(activations)
    print(f'  PCA to {n_pca}d: explained variance = {pca.explained_variance_ratio_.sum():.3f}')

    domain_list = sorted(set(domain_labels))
    n_cca = 5  # Much fewer than sample count

    all_cca_projections = []
    for i, d1 in enumerate(domain_list):
        for d2 in domain_list[i+1:]:
            X_d1_by_shape = {}
            X_d2_by_shape = {}
            for shape in SHAPES:
                mask_d1 = (domain_labels == d1) & (shape_labels == shape)
                mask_d2 = (domain_labels == d2) & (shape_labels == shape)
                X_d1_by_shape[shape] = X_pca[mask_d1]
                X_d2_by_shape[shape] = X_pca[mask_d2]

            X_paired_d1 = np.vstack([X_d1_by_shape[s] for s in SHAPES])
            X_paired_d2 = np.vstack([X_d2_by_shape[s] for s in SHAPES])

            cca = CCA(n_components=n_cca, max_iter=2000)
            cca.fit(X_paired_d1, X_paired_d2)
            X_c1, X_c2 = cca.transform(X_paired_d1, X_paired_d2)
            corrs = [np.corrcoef(X_c1[:, k], X_c2[:, k])[0, 1] for k in range(n_cca)]
            print(f'  {d1:8s} <-> {d2:8s}: corrs = {[f"{c:.3f}" for c in corrs]}')

            all_cca_projections.append({
                'domains': (d1, d2), 'cca': cca, 'correlations': corrs,
            })

    # Build structural subspace from consensus CCA directions
    all_weights = []
    for proj in all_cca_projections:
        all_weights.append(proj['cca'].x_weights_)
        all_weights.append(proj['cca'].y_weights_)

    W = np.vstack([w.T for w in all_weights])
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    n_structural = 5
    structural_basis_pca = Vt[:n_structural]
    X_structural = X_pca @ structural_basis_pca.T
    X_residual = X_pca - X_pca @ structural_basis_pca.T @ structural_basis_pca

    print(f'\n  Structural subspace: {n_structural}d')
    print(f'  SV spectrum: {[f"{s:.2f}" for s in S[:10]]}')

    s_domain, _ = classify(X_structural, domain_labels)
    s_shape, _ = classify(X_structural, shape_labels)
    r_domain, _ = classify(X_residual, domain_labels)
    r_shape, _ = classify(X_residual, shape_labels)

    print(f'\n  Structural: domain={s_domain:.3f}, shape={s_shape:.3f}')
    print(f'  Residual:   domain={r_domain:.3f}, shape={r_shape:.3f}')
    print(f'  Separation: structural={s_shape-s_domain:+.3f}, residual={r_domain-r_shape:+.3f}')

    return all_cca_projections


# ============================================================
# Approach B: INLP (Iterative Null-space Projection)
# ============================================================

def approach_b_inlp(activations, shape_labels, domain_labels):
    """Iteratively remove domain-predictive directions; test shape survival."""
    print('\n' + '=' * 60)
    print('APPROACH B: INLP (ITERATIVE DOMAIN ERASURE)')
    print('=' * 60)

    X = activations.copy()
    le = LabelEncoder()
    y_domain = le.fit_transform(domain_labels)
    n_classes = len(le.classes_)

    print(f'\n  {"Iter":>4s}  {"Domain acc":>10s}  {"Shape acc":>10s}  {"Dims removed":>12s}')
    print(f'  {"----":>4s}  {"----------":>10s}  {"----------":>10s}  {"------------":>12s}')

    # Track shape accuracy as we remove domain directions
    removed_directions = []
    results = []

    for iteration in range(30):
        d_acc, d_std = classify(X, domain_labels)
        s_acc, s_std = classify(X, shape_labels)
        results.append({
            'iteration': iteration,
            'domain_acc': float(d_acc),
            'shape_acc': float(s_acc),
            'dims_removed': len(removed_directions),
        })
        print(f'  {iteration:4d}  {d_acc:10.3f}  {s_acc:10.3f}  {len(removed_directions):12d}')

        # Stop if domain is near chance
        if d_acc < 0.35:
            print(f'\n  Domain accuracy near chance ({1/n_classes:.3f}). Stopping.')
            break

        # Train domain classifier, extract weight directions
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X, domain_labels)

        # Weight matrix: (n_classes, n_features) — each row is a domain direction
        W = clf.coef_  # (n_classes, hidden_dim)

        # Project out the most discriminative direction (first principal component of W)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        direction = Vt[0]  # Most discriminative direction
        direction = direction / np.linalg.norm(direction)

        # Project X onto null space of this direction
        X = X - X @ np.outer(direction, direction)
        removed_directions.append(direction)

    # Final assessment
    d_acc, _ = classify(X, domain_labels)
    s_acc, _ = classify(X, shape_labels)
    results.append({
        'iteration': len(removed_directions),
        'domain_acc': float(d_acc),
        'shape_acc': float(s_acc),
        'dims_removed': len(removed_directions),
    })

    print(f'\n  Final after {len(removed_directions)} directions removed:')
    print(f'    Domain: {d_acc:.3f}')
    print(f'    Shape:  {s_acc:.3f}')

    # Key metric: how much shape accuracy survived domain erasure?
    initial_shape = results[0]['shape_acc']
    shape_survival = s_acc / initial_shape if initial_shape > 0 else 0
    print(f'\n  Shape survival rate: {shape_survival:.1%}')
    print(f'    ({s_acc:.3f} / {initial_shape:.3f})')

    if shape_survival > 0.5 and d_acc < 0.35:
        print(f'  --> STRONG: Shape signal survives domain erasure')
    elif shape_survival > 0.3:
        print(f'  --> MODERATE: Some shape signal independent of domain')
    else:
        print(f'  --> WEAK: Shape and domain heavily entangled')

    # Build domain-erased subspace
    if len(removed_directions) > 0:
        P = np.eye(activations.shape[1])
        for d in removed_directions:
            P = P - np.outer(d, d) @ P  # Careful: accumulate null-space projections
        X_erased = activations @ P.T
    else:
        X_erased = X

    return results, removed_directions, X_erased


# ============================================================
# Approach C: Representational Similarity Analysis
# ============================================================

def approach_c_rsa(activations, shape_labels, domain_labels):
    """Centered Kernel Alignment to measure domain vs shape signal."""
    print('\n' + '=' * 60)
    print('APPROACH C: REPRESENTATIONAL SIMILARITY ANALYSIS')
    print('=' * 60)

    N = len(activations)

    # Activation similarity matrix (cosine)
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    X_normed = activations / (norms + 1e-10)
    K_act = X_normed @ X_normed.T  # (N, N) cosine similarity

    # Domain similarity matrix (1 if same domain, 0 otherwise)
    K_domain = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K_domain[i, j] = 1.0 if domain_labels[i] == domain_labels[j] else 0.0

    # Shape similarity matrix
    K_shape = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K_shape[i, j] = 1.0 if shape_labels[i] == shape_labels[j] else 0.0

    # Centered Kernel Alignment
    def center_kernel(K):
        N = K.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        return H @ K @ H

    def cka(K1, K2):
        K1c = center_kernel(K1)
        K2c = center_kernel(K2)
        hsic = np.sum(K1c * K2c)
        norm1 = np.sqrt(np.sum(K1c * K1c))
        norm2 = np.sqrt(np.sum(K2c * K2c))
        return hsic / (norm1 * norm2 + 1e-10)

    cka_domain = cka(K_act, K_domain)
    cka_shape = cka(K_act, K_shape)

    print(f'\n  CKA(activations, domain): {cka_domain:.4f}')
    print(f'  CKA(activations, shape):  {cka_shape:.4f}')
    print(f'  Ratio (domain/shape):     {cka_domain/(cka_shape+1e-10):.2f}')

    if cka_domain > cka_shape * 2:
        print(f'  --> Domain signal dominates (>2x shape)')
    elif cka_domain > cka_shape:
        print(f'  --> Domain signal stronger but comparable')
    elif cka_shape > cka_domain:
        print(f'  --> Shape signal stronger than domain (!)')
    print(f'\n  Note: CKA measures overall alignment. Both being high is expected')
    print(f'  since both are real structure. The ratio indicates relative strength.')

    # Within-shape vs within-domain activation similarity
    print(f'\n--- Within-group cosine similarities ---')
    within_domain = []
    within_shape = []
    between_all = []
    within_both = []
    within_neither = []

    for i in range(N):
        for j in range(i+1, N):
            sim = K_act[i, j]
            same_domain = domain_labels[i] == domain_labels[j]
            same_shape = shape_labels[i] == shape_labels[j]

            if same_domain and same_shape:
                within_both.append(sim)
            elif same_domain and not same_shape:
                within_domain.append(sim)
            elif same_shape and not same_domain:
                within_shape.append(sim)
            else:
                within_neither.append(sim)

    print(f'  Same domain + same shape:  {np.mean(within_both):.4f} (n={len(within_both)})')
    print(f'  Same domain, diff shape:   {np.mean(within_domain):.4f} (n={len(within_domain)})')
    print(f'  Diff domain, same shape:   {np.mean(within_shape):.4f} (n={len(within_shape)})')
    print(f'  Diff domain + diff shape:  {np.mean(within_neither):.4f} (n={len(within_neither)})')

    domain_effect = np.mean(within_domain) - np.mean(within_neither)
    shape_effect = np.mean(within_shape) - np.mean(within_neither)
    interaction = np.mean(within_both) - np.mean(within_domain) - np.mean(within_shape) + np.mean(within_neither)

    print(f'\n  Domain effect (same domain vs neither): {domain_effect:+.4f}')
    print(f'  Shape effect (same shape vs neither):   {shape_effect:+.4f}')
    print(f'  Interaction:                            {interaction:+.4f}')
    print(f'  Domain/Shape effect ratio:              {domain_effect/(shape_effect+1e-10):.2f}')

    return {
        'cka_domain': float(cka_domain),
        'cka_shape': float(cka_shape),
        'within_both': float(np.mean(within_both)),
        'within_domain_only': float(np.mean(within_domain)),
        'within_shape_only': float(np.mean(within_shape)),
        'within_neither': float(np.mean(within_neither)),
        'domain_effect': float(domain_effect),
        'shape_effect': float(shape_effect),
        'interaction': float(interaction),
    }


# ============================================================
# Main
# ============================================================

def main():
    print('Paper 4 Step 3: Improved Decomposition')
    activations, shape_labels, domain_labels = load_data()
    print(f'Loaded: {activations.shape[0]} probes, {activations.shape[1]}d')

    # Baseline
    print('\n--- Baseline classifiers ---')
    d_acc, d_std = classify(activations, domain_labels)
    s_acc, s_std = classify(activations, shape_labels)
    print(f'  Domain: {d_acc:.3f} +/- {d_std:.3f}')
    print(f'  Shape:  {s_acc:.3f} +/- {s_std:.3f}')

    # Run all three approaches
    cca_results = approach_a_cca(activations, shape_labels, domain_labels)
    inlp_results, removed_dirs, X_erased = approach_b_inlp(activations, shape_labels, domain_labels)
    rsa_results = approach_c_rsa(activations, shape_labels, domain_labels)

    # Save all results
    out_dir = Path('results/paper4')
    with open(out_dir / 'step3_results.json', 'w') as f:
        json.dump({
            'baseline_domain': float(d_acc),
            'baseline_shape': float(s_acc),
            'inlp': inlp_results,
            'rsa': rsa_results,
        }, f, indent=2)
    np.save(out_dir / 'domain_erased_activations.npy', X_erased)
    if len(removed_dirs) > 0:
        np.save(out_dir / 'removed_domain_directions.npy', np.array(removed_dirs))

    print(f'\n  Results saved to {out_dir}/')

    # Final summary
    print('\n' + '=' * 60)
    print('OVERALL ASSESSMENT')
    print('=' * 60)
    final_inlp = inlp_results[-1] if inlp_results else {}
    print(f'  After domain erasure (INLP):')
    print(f'    Domain acc: {final_inlp.get("domain_acc", "?"):.3f}')
    print(f'    Shape acc:  {final_inlp.get("shape_acc", "?"):.3f}')
    print(f'  RSA domain/shape effect ratio: {rsa_results["domain_effect"]/(rsa_results["shape_effect"]+1e-10):.2f}')
    print(f'  CKA domain: {rsa_results["cka_domain"]:.4f}, shape: {rsa_results["cka_shape"]:.4f}')


if __name__ == '__main__':
    main()
