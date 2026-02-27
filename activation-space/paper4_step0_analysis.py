"""Paper 4 Step 0+1: TDA sanity check + domain confound measurement.

Loads Qwen2.5-3B, runs 200 domain probes, captures per-probe activation
vectors (200 x 2048), then:
  Step 0: TDA — persistent homology per domain vs combined, before/after
          naive domain masking (project out domain centroid direction)
  Step 1: Confound check — train linear domain classifier, report accuracy

Usage: python3 paper4_step0_analysis.py
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from paper3_constellation import DOMAINS, DOMAIN_PROBES, PROBES_PER_DOMAIN

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ============================================================
# Step 0a: Capture activations
# ============================================================

def capture_activations(model, tokenizer, probes_by_domain, max_length=128):
    """Forward pass all probes, return (N, hidden_dim) activation matrix."""
    all_texts = []
    all_labels = []
    for domain in DOMAINS:
        if domain not in probes_by_domain:
            continue
        texts = probes_by_domain[domain]
        all_texts.extend(texts)
        all_labels.extend([domain] * len(texts))

    encoded = tokenizer(
        all_texts, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    model.eval()
    device = next(model.parameters()).device
    activations = []
    batch_size = 4  # Conservative for 3B on MPS

    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_ids = encoded['input_ids'][i:i+batch_size].to(device)
            batch_mask = encoded['attention_mask'][i:i+batch_size].to(device)
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask,
                           output_hidden_states=True)
            # Final hidden state
            hidden = outputs.hidden_states[-1].float()  # (B, seq_len, hidden_dim)
            # Use mean of non-padding tokens
            mask_expanded = batch_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            activations.append(pooled.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f'    batch {i//batch_size + 1}/{(len(all_texts) + batch_size - 1)//batch_size}')

    activations = np.concatenate(activations, axis=0)
    return activations, all_labels, all_texts


# ============================================================
# Step 0b: TDA — persistent homology
# ============================================================

def run_tda(activations, labels, domains_to_use):
    """Compute persistent homology per domain and check Betti number similarity."""
    try:
        from ripser import ripser
        from persim import sliced_wasserstein
        has_tda = True
    except ImportError:
        has_tda = False

    domain_indices = {}
    for i, label in enumerate(labels):
        if label in domains_to_use:
            domain_indices.setdefault(label, []).append(i)

    # Compute domain centroids
    centroids = {}
    for domain, indices in domain_indices.items():
        centroids[domain] = activations[indices].mean(axis=0)
    global_centroid = activations.mean(axis=0)

    # Naive domain masking: project out domain centroid direction from global
    masked_activations = activations.copy()
    for i, label in enumerate(labels):
        if label in domains_to_use:
            direction = centroids[label] - global_centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
                projection = np.dot(masked_activations[i], direction)
                masked_activations[i] = masked_activations[i] - projection * direction

    print('\n' + '=' * 60)
    print('STEP 0: TDA SANITY CHECK')
    print('=' * 60)

    # Basic geometry stats before/after masking
    print('\n--- Pairwise centroid distances (cosine) ---')
    domain_list = sorted(domains_to_use)
    for phase_name, act_matrix in [('Raw', activations), ('Domain-masked', masked_activations)]:
        print(f'\n  {phase_name}:')
        phase_centroids = {}
        for domain in domain_list:
            idx = domain_indices[domain]
            phase_centroids[domain] = act_matrix[idx].mean(axis=0)

        for i, d1 in enumerate(domain_list):
            for d2 in domain_list[i+1:]:
                c1, c2 = phase_centroids[d1], phase_centroids[d2]
                cos_sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10)
                print(f'    {d1:8s} vs {d2:8s}: cosine={cos_sim:.4f}')

    # Within-domain vs between-domain distance ratio
    for phase_name, act_matrix in [('Raw', activations), ('Domain-masked', masked_activations)]:
        within_dists = []
        between_dists = []
        for d1 in domain_list:
            pts1 = act_matrix[domain_indices[d1]]
            # Within-domain: pairwise distances
            for i in range(len(pts1)):
                for j in range(i+1, len(pts1)):
                    within_dists.append(np.linalg.norm(pts1[i] - pts1[j]))
            # Between-domain
            for d2 in domain_list:
                if d2 <= d1:
                    continue
                pts2 = act_matrix[domain_indices[d2]]
                for p1 in pts1[:10]:  # sample for speed
                    for p2 in pts2[:10]:
                        between_dists.append(np.linalg.norm(p1 - p2))

        ratio = np.mean(between_dists) / (np.mean(within_dists) + 1e-10)
        print(f'\n  {phase_name} between/within distance ratio: {ratio:.3f}')
        print(f'    (>1 = domains are separable, ~1 = mixed, <1 = inverted)')

    if has_tda:
        print('\n--- Persistent homology (H0, H1) ---')
        for phase_name, act_matrix in [('Raw', activations), ('Domain-masked', masked_activations)]:
            print(f'\n  {phase_name}:')
            # PCA to 50 dims for computational tractability
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, act_matrix.shape[1]))
            reduced = pca.fit_transform(act_matrix)

            for domain in domain_list:
                idx = domain_indices[domain]
                pts = reduced[idx]
                result = ripser(pts, maxdim=1)
                dgms = result['dgms']
                h0 = dgms[0]
                h1 = dgms[1] if len(dgms) > 1 else np.array([]).reshape(0, 2)
                # Count significant features (persistence > median)
                h0_pers = h0[np.isfinite(h0[:, 1]), 1] - h0[np.isfinite(h0[:, 1]), 0]
                h1_pers = h1[:, 1] - h1[:, 0] if len(h1) > 0 else np.array([])
                print(f'    {domain:8s}: H0 components={len(h0)}, '
                      f'H1 loops={len(h1)}, '
                      f'H1 mean persistence={np.mean(h1_pers):.4f}' if len(h1_pers) > 0
                      else f'    {domain:8s}: H0 components={len(h0)}, H1 loops=0')
    else:
        print('\n  [ripser not installed — skipping persistent homology]')
        print('  Install with: pip install ripser persim')

    return masked_activations


# ============================================================
# Step 1: Confound check — domain classifier
# ============================================================

def run_confound_check(activations, labels, domains_to_use):
    """Train linear domain classifier, report cross-validated accuracy."""
    print('\n' + '=' * 60)
    print('STEP 1: CONFOUND CHECK')
    print('=' * 60)

    # Filter to domains of interest
    mask = [l in domains_to_use for l in labels]
    X = activations[mask]
    y = np.array([l for l in labels if l in domains_to_use])

    # Linear domain classifier with 5-fold CV
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f'\n  Domain classifier (logistic regression, 5-fold CV):')
    print(f'    Accuracy: {scores.mean():.3f} ± {scores.std():.3f}')
    print(f'    Per-fold: {[f"{s:.3f}" for s in scores]}')
    print(f'    Chance: {1/len(domains_to_use):.3f}')

    if scores.mean() > 0.9:
        print(f'\n  --> Domains are highly separable in activation space (good)')
    elif scores.mean() > 0.5:
        print(f'\n  --> Domains are partially separable')
    else:
        print(f'\n  --> Domains are not separable (bad — may lack structure)')

    # Check per-domain accuracy
    clf.fit(X, y)
    per_domain = {}
    for domain in sorted(domains_to_use):
        domain_mask = y == domain
        pred = clf.predict(X[domain_mask])
        acc = (pred == domain).mean()
        per_domain[domain] = acc
        print(f'    {domain:8s} accuracy: {acc:.3f}')

    # PCA visualization data
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    print(f'\n  PCA explained variance (3 components): {pca.explained_variance_ratio_}')
    print(f'    Total: {pca.explained_variance_ratio_.sum():.3f}')

    return scores.mean()


# ============================================================
# Main
# ============================================================

def main():
    print('Paper 4 Step 0+1: TDA + Confound Check')
    print(f'Model: Qwen/Qwen2.5-3B (2048-dim hidden states)')

    # Use only the 4 specialist domains (exclude 'general')
    domains_to_use = {'medical', 'legal', 'code', 'science'}

    # Load model
    print('\nLoading Qwen2.5-3B...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B', dtype=torch.float16,
        device_map='cpu', low_cpu_mem_usage=True,
    )
    print(f'  Hidden dim: {model.config.hidden_size}')
    print(f'  Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B')

    # Filter probes to specialist domains only
    probes = {d: DOMAIN_PROBES[d] for d in DOMAIN_PROBES if d in domains_to_use}
    total_probes = sum(len(v) for v in probes.values())
    print(f'  Probes: {total_probes} across {len(probes)} domains')

    # Capture activations
    print('\nCapturing activations...')
    activations, labels, texts = capture_activations(model, tokenizer, probes)
    print(f'  Activation matrix: {activations.shape}')

    # Free model memory
    del model
    torch.mps.empty_cache() if hasattr(torch, 'mps') else None

    # Save activations for downstream use
    out_dir = Path('results/paper4')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'qwen3b_activations.npy', activations)
    with open(out_dir / 'qwen3b_labels.json', 'w') as f:
        json.dump({'labels': labels, 'texts': texts}, f)
    print(f'  Saved to {out_dir}/')

    # Run analyses
    masked = run_tda(activations, labels, domains_to_use)
    accuracy = run_confound_check(activations, labels, domains_to_use)

    # Also run confound check on domain-masked activations
    print('\n' + '=' * 60)
    print('STEP 1b: CONFOUND CHECK ON DOMAIN-MASKED ACTIVATIONS')
    print('=' * 60)
    masked_acc = run_confound_check(masked, labels, domains_to_use)

    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Raw domain classifier accuracy:    {accuracy:.3f}')
    print(f'  Masked domain classifier accuracy:  {masked_acc:.3f}')
    print(f'  Drop from masking:                  {accuracy - masked_acc:+.3f}')
    if masked_acc < 0.35:
        print(f'  --> Domain masking is effective (near chance={1/len(domains_to_use):.3f})')
    else:
        print(f'  --> Domain signal persists after naive masking — need better decomposition')


if __name__ == '__main__':
    main()
