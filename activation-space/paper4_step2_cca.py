"""Paper 4 Step 2+3: Capture shape-labeled activations + CCA decomposition.

Loads Qwen2.5-3B, runs 160 shape-labeled probes (4 shapes x 4 domains x 10),
captures per-probe activation vectors (160 x 2048), then:
  Step 2: Capture activations with shape+domain labels
  Step 3: CCA decomposition — find structural subspace vs domain subspace
  Step 3b: Evaluate decomposition quality

Usage: KMP_DUPLICATE_LIB_OK=TRUE python3 paper4_step2_cca.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paper4_probes import SHAPES, DOMAINS, SHAPE_PROBES, get_all_probes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Step 2: Capture activations
# ============================================================

def capture_activations(model, tokenizer, probes, max_length=128, batch_size=4):
    """Forward pass all probes, return (N, hidden_dim) activation matrix."""
    texts = [p[2] for p in probes]
    encoded = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    model.eval()
    device = next(model.parameters()).device
    activations = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_ids = encoded['input_ids'][i:i+batch_size].to(device)
            batch_mask = encoded['attention_mask'][i:i+batch_size].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                           output_hidden_states=True)
            hidden = outputs.hidden_states[-1].float()
            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            activations.append(pooled.cpu().numpy())
            batch_num = i // batch_size + 1
            if batch_num % 10 == 1 or batch_num == n_batches:
                print(f'    batch {batch_num}/{n_batches}')

    return np.concatenate(activations, axis=0)


# ============================================================
# Step 3: CCA decomposition
# ============================================================

def run_classifiers(X, shape_labels, domain_labels, prefix=""):
    """Run shape and domain classifiers, return accuracies."""
    clf_domain = LogisticRegression(max_iter=1000, solver='lbfgs')
    domain_scores = cross_val_score(clf_domain, X, domain_labels, cv=5, scoring='accuracy')

    clf_shape = LogisticRegression(max_iter=1000, solver='lbfgs')
    shape_scores = cross_val_score(clf_shape, X, shape_labels, cv=5, scoring='accuracy')

    print(f'  {prefix}Domain classifier:  {domain_scores.mean():.3f} +/- {domain_scores.std():.3f}  (chance={1/len(set(domain_labels)):.3f})')
    print(f'  {prefix}Shape classifier:   {shape_scores.mean():.3f} +/- {shape_scores.std():.3f}  (chance={1/len(set(shape_labels)):.3f})')

    return domain_scores.mean(), shape_scores.mean()


def run_cca_decomposition(activations, probes):
    """Use CCA to find shared structural subspace across domain pairs."""
    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])

    print('\n' + '=' * 60)
    print('STEP 3: CCA DECOMPOSITION')
    print('=' * 60)

    # --- Baseline classifiers on raw activations ---
    print('\n--- Baseline classifiers (raw activations) ---')
    raw_domain_acc, raw_shape_acc = run_classifiers(
        activations, shape_labels, domain_labels, prefix="Raw ")

    # --- PCA reduction for stability ---
    n_components = 50
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(activations)
    print(f'\n  PCA to {n_components}d: explained variance = {pca.explained_variance_ratio_.sum():.3f}')

    # --- Pairwise CCA across domains ---
    # For each pair of domains, find dimensions that correlate across
    # same-shape probes. These are candidate structural dimensions.
    print('\n--- Pairwise CCA (same-shape probes across domain pairs) ---')

    domain_list = sorted(set(domain_labels))
    cca_components = min(10, n_components)

    # Build paired observations: for each shape, pair domain A and domain B activations
    all_cca_projections = []

    for i, d1 in enumerate(domain_list):
        for d2 in domain_list[i+1:]:
            # Get same-shape pairs across these two domains
            X_d1_by_shape = {}
            X_d2_by_shape = {}
            for shape in SHAPES:
                mask_d1 = (domain_labels == d1) & (shape_labels == shape)
                mask_d2 = (domain_labels == d2) & (shape_labels == shape)
                X_d1_by_shape[shape] = X_pca[mask_d1]
                X_d2_by_shape[shape] = X_pca[mask_d2]

            # Stack: rows are matched by shape (probe index within shape)
            X_paired_d1 = np.vstack([X_d1_by_shape[s] for s in SHAPES])
            X_paired_d2 = np.vstack([X_d2_by_shape[s] for s in SHAPES])

            # CCA: find projections that maximize correlation between domains
            n_cca = min(cca_components, X_paired_d1.shape[0] - 1)
            cca = CCA(n_components=n_cca, max_iter=1000)
            cca.fit(X_paired_d1, X_paired_d2)

            X_c1, X_c2 = cca.transform(X_paired_d1, X_paired_d2)

            # Canonical correlations
            corrs = [np.corrcoef(X_c1[:, k], X_c2[:, k])[0, 1] for k in range(n_cca)]
            print(f'  {d1:8s} <-> {d2:8s}: top corrs = {[f"{c:.3f}" for c in corrs[:5]]}')

            all_cca_projections.append({
                'domains': (d1, d2),
                'cca': cca,
                'correlations': corrs,
            })

    # --- Multi-set CCA approximation via shared PCA subspace ---
    # Project all activations into the mean CCA direction across all pairs
    print('\n--- Structural subspace construction ---')

    # Collect CCA weight vectors (domain 1 side) from all pairs
    all_weights = []
    for proj in all_cca_projections:
        # x_weights_ shape: (n_features, n_components) - maps from PCA space to CCA space
        all_weights.append(proj['cca'].x_weights_)
        all_weights.append(proj['cca'].y_weights_)

    # Stack and SVD to find the consensus structural directions
    W = np.vstack([w.T for w in all_weights])  # (n_pairs * 2 * n_cca, n_pca)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # Top-k singular vectors of CCA weights = structural subspace in PCA space
    # Then compose with PCA to get back to activation space
    n_structural = 10
    structural_basis_pca = Vt[:n_structural]  # (n_structural, n_pca)
    structural_basis_full = structural_basis_pca @ pca.components_  # (n_structural, hidden_dim)

    # Project activations into structural and residual subspaces
    X_structural = activations @ structural_basis_full.T  # (N, n_structural)

    # Residual: remove structural projection from PCA space
    X_structural_in_pca = X_pca @ structural_basis_pca.T @ structural_basis_pca  # (N, n_pca)
    X_residual_pca = X_pca - X_structural_in_pca

    print(f'  Structural subspace: {n_structural} dimensions')
    print(f'  Singular value spectrum: {[f"{s:.2f}" for s in S[:15]]}')
    print(f'  Top-10 capture {S[:10].sum()/S.sum()*100:.1f}% of CCA weight variance')

    # --- Classifiers on decomposed subspaces ---
    print('\n--- Classifiers on STRUCTURAL subspace ---')
    struct_domain_acc, struct_shape_acc = run_classifiers(
        X_structural, shape_labels, domain_labels, prefix="Structural ")

    print('\n--- Classifiers on RESIDUAL subspace ---')
    resid_domain_acc, resid_shape_acc = run_classifiers(
        X_residual_pca, shape_labels, domain_labels, prefix="Residual ")

    # --- Ideal result: structural subspace has high shape accuracy, low domain accuracy
    #     and residual subspace has high domain accuracy, low shape accuracy ---
    print('\n' + '=' * 60)
    print('DECOMPOSITION QUALITY SUMMARY')
    print('=' * 60)
    print(f'\n  {"Subspace":<20s} {"Domain acc":>12s} {"Shape acc":>12s} {"Interpretation":<30s}')
    print(f'  {"-"*20:<20s} {"-"*12:>12s} {"-"*12:>12s} {"-"*30:<30s}')
    print(f'  {"Raw (baseline)":<20s} {raw_domain_acc:>12.3f} {raw_shape_acc:>12.3f} {"both encoded":<30s}')
    print(f'  {"Structural (CCA)":<20s} {struct_domain_acc:>12.3f} {struct_shape_acc:>12.3f} {"want: low domain, high shape":<30s}')
    print(f'  {"Residual":<20s} {resid_domain_acc:>12.3f} {resid_shape_acc:>12.3f} {"want: high domain, low shape":<30s}')
    print(f'  {"Chance":<20s} {0.250:>12.3f} {0.250:>12.3f}')

    # Separation index: how well did the decomposition separate domain from shape?
    # Perfect separation: structural has shape=1.0, domain=0.25; residual has domain=1.0, shape=0.25
    structural_separation = struct_shape_acc - struct_domain_acc
    residual_separation = resid_domain_acc - resid_shape_acc
    total_separation = structural_separation + residual_separation

    print(f'\n  Structural separation (shape - domain): {structural_separation:+.3f}')
    print(f'  Residual separation (domain - shape):   {residual_separation:+.3f}')
    print(f'  Total separation index:                 {total_separation:+.3f}')
    print(f'  (Ideal: +1.5 = perfect; 0 = no separation; negative = inverted)')

    if total_separation > 0.3:
        print(f'\n  --> PROMISING: CCA found partially separable domain/structure components')
    elif total_separation > 0.0:
        print(f'\n  --> WEAK: Some separation but heavily entangled')
    else:
        print(f'\n  --> NEGATIVE: Decomposition did not separate domain from structure')

    # Save results
    results = {
        'raw_domain_acc': float(raw_domain_acc),
        'raw_shape_acc': float(raw_shape_acc),
        'structural_domain_acc': float(struct_domain_acc),
        'structural_shape_acc': float(struct_shape_acc),
        'residual_domain_acc': float(resid_domain_acc),
        'residual_shape_acc': float(resid_shape_acc),
        'total_separation_index': float(total_separation),
        'n_structural_dims': n_structural,
        'n_pca_dims': n_components,
        'cca_correlations': {
            f'{p["domains"][0]}_{p["domains"][1]}': [float(c) for c in p['correlations']]
            for p in all_cca_projections
        },
        'singular_values': [float(s) for s in S[:20]],
    }

    return results, X_structural, X_residual_pca, structural_basis_full


# ============================================================
# Main
# ============================================================

def main():
    print('Paper 4 Step 2+3: Shape-Labeled Activations + CCA Decomposition')
    print(f'Model: Qwen/Qwen2.5-3B')

    probes = get_all_probes()
    print(f'Probes: {len(probes)} ({len(SHAPES)} shapes x {len(DOMAINS)} domains x 10)')

    # Load model
    print('\nLoading Qwen2.5-3B...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B', dtype=torch.float16,
        device_map='cpu', low_cpu_mem_usage=True,
    )
    print(f'  Hidden dim: {model.config.hidden_size}')

    # Capture activations
    print('\nCapturing activations...')
    activations = capture_activations(model, tokenizer, probes)
    print(f'  Activation matrix: {activations.shape}')

    # Free model memory
    del model, tokenizer
    import gc; gc.collect()

    # Save raw activations
    out_dir = Path('results/paper4')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'shape_activations.npy', activations)
    with open(out_dir / 'shape_labels.json', 'w') as f:
        json.dump({
            'probes': [(s, d, t) for s, d, t in probes],
            'shapes': SHAPES,
            'domains': DOMAINS,
        }, f, indent=2)
    print(f'  Saved activations to {out_dir}/')

    # Run CCA decomposition
    results, X_struct, X_resid, struct_basis = run_cca_decomposition(activations, probes)

    # Save decomposition results
    with open(out_dir / 'cca_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.save(out_dir / 'structural_basis.npy', struct_basis)
    np.save(out_dir / 'structural_projections.npy', X_struct)
    np.save(out_dir / 'residual_projections.npy', X_resid)
    print(f'\n  All results saved to {out_dir}/')


if __name__ == '__main__':
    main()
