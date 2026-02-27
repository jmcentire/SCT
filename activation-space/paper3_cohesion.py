#!/usr/bin/env python3
"""Paper 3: Category Cohesion Analysis.

Computes activation-space cohesion metrics for each domain's probe set
to explain why some categories (medical) fail at retrieval while others
(legal) succeed. Proposes a threshold metric for category granularity.

Metrics computed per domain:
  1. Cohesion: mean pairwise cosine similarity within category
  2. Centroid discriminability: ratio of intra-class distance to nearest
     inter-class centroid distance
  3. Effective rank: entropy of normalized singular value spectrum
  4. Intra-class variance: trace of covariance matrix

Also splits "medical" into subcategories and recomputes metrics to show
finer-grained categories have higher cohesion.

Usage:
  python paper3_cohesion.py \
      --model Qwen/Qwen2.5-1.5B \
      --generalist-ckpt results/paper3_scale/Qwen_Qwen2.5-1.5B/seed42/step_02000.pt \
      --specialist-dir results/paper3_scale/Qwen_Qwen2.5-1.5B/seed42/two_stage/specialists \
      --output-dir results/paper3_scale/Qwen_Qwen2.5-1.5B/seed42/cohesion \
      --device cuda
"""
import os, sys, json, gc, argparse, time
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(__file__) or '.')
from src.fingerprint.capture import capture_at_checkpoint
from paper3_constellation import (
    DOMAINS, DOMAIN_PROBES, PROBES_PER_DOMAIN,
    tokenize_domain_probes, _set_model_cls, _get_model_cls,
)

# ============================================================
# Medical subcategory assignments (0-indexed into DOMAIN_PROBES['medical'])
# ============================================================

MEDICAL_SUBCATEGORIES = {
    'pharmacology': [
        1,   # Metformin
        2,   # SSRIs mechanism
        13,  # Warfarin pharmacokinetics
        21,  # Beta-blockers
        27,  # Pain management / WHO analgesic ladder
        35,  # Perioperative anticoagulation
    ],
    'diagnostics_imaging': [
        3,   # MRI lesion
        8,   # Glasgow Coma Scale
        10,  # Pulmonary function tests
        17,  # ECG / MI
        20,  # CSF analysis
        23,  # HbA1c
        33,  # Genetic testing / BRCA
    ],
    'clinical_surgical': [
        0,   # Acute respiratory distress
        4,   # Hypertension treatment
        12,  # Bowel obstruction / surgery
        24,  # DVT risk factors
        30,  # Sepsis
        37,  # Wound healing
        38,  # Liver cirrhosis
    ],
    'pathophysiology': [
        5,   # Alzheimer's pathophysiology
        6,   # Cardiac catheterization / stenosis
        7,   # IgE allergic reactions
        9,   # Antibiotic resistance
        11,  # HPA axis
        14,  # Lung cancer survival
        15,  # CKD staging
        16,  # Blood-brain barrier
        25,  # RAAS
        26,  # Autoimmune diseases
        34,  # Complement system
        36,  # Rheumatoid vs osteoarthritis
        39,  # Neuroplasticity
    ],
    'public_health': [
        18,  # Vaccination
        19,  # Differential diagnosis chest pain
        22,  # WHO brain tumor classification
        28,  # Clinical trial phases
        29,  # Osteoporosis diagnosis
        31,  # Cranial nerves
        32,  # Insulin resistance / metabolic syndrome
    ],
}

# Verify all 40 probes are assigned exactly once
_all_assigned = []
for _indices in MEDICAL_SUBCATEGORIES.values():
    _all_assigned.extend(_indices)
assert sorted(_all_assigned) == list(range(40)), \
    f"Medical subcategory assignments don't cover all 40 probes: {sorted(_all_assigned)}"


# ============================================================
# Metric computations
# ============================================================

def cosine_sim_matrix(X):
    """Compute pairwise cosine similarity matrix for rows of X.

    Args:
        X: np.ndarray of shape (n, d)

    Returns:
        np.ndarray of shape (n, n) with cosine similarities
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_norm = X / norms
    return X_norm @ X_norm.T


def compute_cohesion(X):
    """Mean pairwise cosine similarity (excluding self-similarity).

    Args:
        X: np.ndarray of shape (n, d)

    Returns:
        float: mean off-diagonal cosine similarity
    """
    if len(X) < 2:
        return 1.0
    sim = cosine_sim_matrix(X)
    n = len(sim)
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim[mask].mean())


def compute_cohesion_std(X):
    """Standard deviation of pairwise cosine similarities."""
    if len(X) < 2:
        return 0.0
    sim = cosine_sim_matrix(X)
    n = len(sim)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim[mask].std())


def compute_effective_rank(X):
    """Effective rank via entropy of normalized singular value spectrum.

    Higher effective rank = fingerprints spread across more dimensions = dispersed.

    Args:
        X: np.ndarray of shape (n, d)

    Returns:
        float: effective rank (1 = all variance on one axis, n = uniformly spread)
    """
    if len(X) < 2:
        return 1.0
    # Center the data
    X_centered = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X_centered, compute_uv=False)
    # Normalize singular values to probability distribution
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def compute_intra_variance(X):
    """Trace of covariance matrix (total variance).

    Args:
        X: np.ndarray of shape (n, d)

    Returns:
        float: trace of covariance (sum of eigenvalues)
    """
    if len(X) < 2:
        return 0.0
    X_centered = X - X.mean(axis=0, keepdims=True)
    # trace(cov) = mean of squared distances from centroid
    return float(np.mean(np.sum(X_centered ** 2, axis=1)))


def compute_centroid_discriminability(X, centroid, other_centroids):
    """Ratio of intra-class spread to nearest inter-class centroid distance.

    Lower = better (tight cluster far from others).

    Args:
        X: np.ndarray of shape (n, d) — probe fingerprints for this category
        centroid: np.ndarray of shape (d,) — centroid of this category
        other_centroids: list of np.ndarray of shape (d,) — centroids of other categories

    Returns:
        float: mean_intra_distance / min_inter_distance
    """
    if len(X) < 2 or len(other_centroids) == 0:
        return 0.0
    # Mean cosine distance from probes to own centroid
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    intra_sims = X_norm @ c_norm
    mean_intra_dist = 1.0 - float(intra_sims.mean())

    # Min cosine distance to any other centroid
    min_inter_dist = float('inf')
    for oc in other_centroids:
        oc_norm = oc / (np.linalg.norm(oc) + 1e-8)
        inter_dist = 1.0 - float(c_norm @ oc_norm)
        min_inter_dist = min(min_inter_dist, inter_dist)

    if min_inter_dist < 1e-8:
        return float('inf')
    return mean_intra_dist / min_inter_dist


def compute_all_metrics(X, centroid=None, other_centroids=None):
    """Compute all cohesion metrics for a set of fingerprints.

    Args:
        X: np.ndarray of shape (n, d)
        centroid: optional np.ndarray of shape (d,)
        other_centroids: optional list of np.ndarray of shape (d,)

    Returns:
        dict of metric_name -> float
    """
    if centroid is None:
        centroid = X.mean(axis=0)
    if other_centroids is None:
        other_centroids = []

    return {
        'cohesion': compute_cohesion(X),
        'cohesion_std': compute_cohesion_std(X),
        'effective_rank': compute_effective_rank(X),
        'intra_variance': compute_intra_variance(X),
        'centroid_discriminability': compute_centroid_discriminability(
            X, centroid, other_centroids),
        'n_probes': len(X),
    }


# ============================================================
# Fingerprint extraction
# ============================================================

def extract_per_probe_fingerprints(model_path, model_cfg, probe_set, device='cpu'):
    """Load a specialist and extract per-probe fingerprints.

    Args:
        model_path: path to specialist.pt checkpoint
        model_cfg: model config for instantiation
        probe_set: tokenized probe set from tokenize_domain_probes()
        device: compute device

    Returns:
        np.ndarray of shape (n_probes, hidden_dim)
    """
    ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
    sd = {k: v.float() for k, v in ckpt['state_dict'].items()}

    model = _get_model_cls()(model_cfg)
    model.load_state_dict(sd)
    model.to(device)

    fp = capture_at_checkpoint(model, probe_set, ['final_hidden'],
                               device=str(device), batch_size=32)

    del model, sd
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return fp['final_hidden']  # (n_probes, hidden_dim)


# ============================================================
# Part A: Domain-level metrics
# ============================================================

def run_part_a(specialist_dir, generalist_ckpt, model_cfg, probe_set, device='cpu'):
    """Compute cohesion metrics for all 5 domains using each specialist's fingerprints."""
    print('\n' + '=' * 60)
    print('PART A: Domain-Level Cohesion Metrics')
    print('=' * 60)

    domain_fingerprints = {}  # domain -> (40, hidden_dim)
    domain_centroids = {}     # domain -> (hidden_dim,)

    for domain in DOMAINS:
        if domain == 'general':
            continue
        spec_path = Path(specialist_dir) / domain / 'specialist.pt'
        if not spec_path.exists():
            print(f'  WARNING: No specialist for {domain}, skipping')
            continue

        print(f'  Extracting fingerprints for {domain} specialist...')
        t0 = time.time()
        all_fps = extract_per_probe_fingerprints(spec_path, model_cfg, probe_set, device)

        # Extract this domain's probes
        d_idx = DOMAINS.index(domain)
        start = d_idx * PROBES_PER_DOMAIN
        end = start + PROBES_PER_DOMAIN
        domain_fps = all_fps[start:end]  # (40, hidden_dim)

        domain_fingerprints[domain] = domain_fps
        domain_centroids[domain] = domain_fps.mean(axis=0)
        print(f'    {domain}: {domain_fps.shape} in {time.time()-t0:.1f}s')

    # Also get generalist fingerprints for all probes
    print(f'  Extracting generalist fingerprints...')
    t0 = time.time()
    gen_ckpt = torch.load(generalist_ckpt, weights_only=False, map_location='cpu')
    gen_sd = {k: v.float() for k, v in gen_ckpt['state_dict'].items()}
    gen_model = _get_model_cls()(model_cfg)
    gen_model.load_state_dict(gen_sd)
    gen_model.to(device)
    gen_fp = capture_at_checkpoint(gen_model, probe_set, ['final_hidden'],
                                   device=str(device), batch_size=32)
    gen_all_fps = gen_fp['final_hidden']
    del gen_model, gen_sd
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'    generalist: {gen_all_fps.shape} in {time.time()-t0:.1f}s')

    # Store generalist per-domain fingerprints
    gen_domain_fps = {}
    gen_domain_centroids = {}
    for domain in DOMAINS:
        if domain == 'general':
            continue
        d_idx = DOMAINS.index(domain)
        start = d_idx * PROBES_PER_DOMAIN
        end = start + PROBES_PER_DOMAIN
        gen_domain_fps[domain] = gen_all_fps[start:end]
        gen_domain_centroids[domain] = gen_all_fps[start:end].mean(axis=0)

    # Compute metrics for each domain
    results = {}
    for domain in DOMAINS:
        if domain == 'general' or domain not in domain_fingerprints:
            continue

        other_centroids = [domain_centroids[d] for d in domain_centroids if d != domain]
        spec_metrics = compute_all_metrics(
            domain_fingerprints[domain], domain_centroids[domain], other_centroids)

        gen_other_centroids = [gen_domain_centroids[d] for d in gen_domain_centroids if d != domain]
        gen_metrics = compute_all_metrics(
            gen_domain_fps[domain], gen_domain_centroids[domain], gen_other_centroids)

        results[domain] = {
            'specialist_metrics': spec_metrics,
            'generalist_metrics': gen_metrics,
        }

        print(f'\n  {domain}:')
        print(f'    Specialist — cohesion: {spec_metrics["cohesion"]:.4f} '
              f'(±{spec_metrics["cohesion_std"]:.4f}), '
              f'eff_rank: {spec_metrics["effective_rank"]:.2f}, '
              f'discrim: {spec_metrics["centroid_discriminability"]:.4f}')
        print(f'    Generalist — cohesion: {gen_metrics["cohesion"]:.4f} '
              f'(±{gen_metrics["cohesion_std"]:.4f}), '
              f'eff_rank: {gen_metrics["effective_rank"]:.2f}, '
              f'discrim: {gen_metrics["centroid_discriminability"]:.4f}')

    return results, domain_fingerprints, domain_centroids, gen_domain_fps, gen_domain_centroids


# ============================================================
# Part B: Medical subcategory analysis
# ============================================================

def run_part_b(domain_fingerprints, domain_centroids, gen_domain_fps, gen_domain_centroids):
    """Split medical into subcategories, compute metrics for each."""
    print('\n' + '=' * 60)
    print('PART B: Medical Subcategory Analysis')
    print('=' * 60)

    if 'medical' not in domain_fingerprints:
        print('  No medical fingerprints available, skipping')
        return {}

    med_fps = domain_fingerprints['medical']  # (40, hidden_dim)
    gen_med_fps = gen_domain_fps['medical']

    # Other domain centroids for discriminability
    spec_other = [domain_centroids[d] for d in domain_centroids if d != 'medical']
    gen_other = [gen_domain_centroids[d] for d in gen_domain_centroids if d != 'medical']

    results = {}
    for subcat, indices in MEDICAL_SUBCATEGORIES.items():
        sub_fps = med_fps[indices]
        sub_centroid = sub_fps.mean(axis=0)
        spec_metrics = compute_all_metrics(sub_fps, sub_centroid, spec_other)

        gen_sub_fps = gen_med_fps[indices]
        gen_sub_centroid = gen_sub_fps.mean(axis=0)
        gen_metrics = compute_all_metrics(gen_sub_fps, gen_sub_centroid, gen_other)

        results[subcat] = {
            'indices': indices,
            'specialist_metrics': spec_metrics,
            'generalist_metrics': gen_metrics,
        }

        print(f'\n  {subcat} (n={len(indices)}):')
        print(f'    Specialist — cohesion: {spec_metrics["cohesion"]:.4f} '
              f'(±{spec_metrics["cohesion_std"]:.4f}), '
              f'eff_rank: {spec_metrics["effective_rank"]:.2f}')
        print(f'    Generalist — cohesion: {gen_metrics["cohesion"]:.4f} '
              f'(±{gen_metrics["cohesion_std"]:.4f}), '
              f'eff_rank: {gen_metrics["effective_rank"]:.2f}')

    # Also report medical-as-whole for comparison
    print(f'\n  medical (whole, n=40):')
    spec_whole = compute_all_metrics(med_fps, med_fps.mean(axis=0), spec_other)
    gen_whole = compute_all_metrics(gen_med_fps, gen_med_fps.mean(axis=0), gen_other)
    print(f'    Specialist — cohesion: {spec_whole["cohesion"]:.4f}, '
          f'eff_rank: {spec_whole["effective_rank"]:.2f}')
    print(f'    Generalist — cohesion: {gen_whole["cohesion"]:.4f}, '
          f'eff_rank: {gen_whole["effective_rank"]:.2f}')

    results['_medical_whole'] = {
        'specialist_metrics': spec_whole,
        'generalist_metrics': gen_whole,
    }

    return results


# ============================================================
# Part C: Synthetic granularity sweep (merge categories)
# ============================================================

def run_part_c(domain_fingerprints, domain_centroids, gen_domain_fps, gen_domain_centroids):
    """Merge domain pairs to create coarser categories, show metric degrades."""
    print('\n' + '=' * 60)
    print('PART C: Synthetic Granularity Sweep')
    print('=' * 60)

    domains = [d for d in DOMAINS if d != 'general' and d in domain_fingerprints]

    results = {}

    # Single domains (baseline)
    for domain in domains:
        fps = domain_fingerprints[domain]
        other = [domain_centroids[d] for d in domains if d != domain]
        metrics = compute_all_metrics(fps, fps.mean(axis=0), other)
        results[domain] = {
            'type': 'single',
            'domains': [domain],
            'specialist_metrics': metrics,
        }

    # All pairs
    for i, d1 in enumerate(domains):
        for d2 in domains[i+1:]:
            merged_fps = np.concatenate([
                domain_fingerprints[d1], domain_fingerprints[d2]], axis=0)
            # "Other" centroids = all domains not in this pair
            others = [domain_centroids[d] for d in domains if d not in (d1, d2)]
            merged_centroid = merged_fps.mean(axis=0)
            metrics = compute_all_metrics(merged_fps, merged_centroid, others)
            key = f'{d1}+{d2}'
            results[key] = {
                'type': 'pair',
                'domains': [d1, d2],
                'specialist_metrics': metrics,
            }
            print(f'  {key}: cohesion={metrics["cohesion"]:.4f}, '
                  f'eff_rank={metrics["effective_rank"]:.2f}')

    # All triples
    for i, d1 in enumerate(domains):
        for j, d2 in enumerate(domains[i+1:], i+1):
            for d3 in domains[j+1:]:
                merged_fps = np.concatenate([
                    domain_fingerprints[d1],
                    domain_fingerprints[d2],
                    domain_fingerprints[d3]], axis=0)
                others = [domain_centroids[d] for d in domains
                          if d not in (d1, d2, d3)]
                merged_centroid = merged_fps.mean(axis=0)
                metrics = compute_all_metrics(merged_fps, merged_centroid, others)
                key = f'{d1}+{d2}+{d3}'
                results[key] = {
                    'type': 'triple',
                    'domains': [d1, d2, d3],
                    'specialist_metrics': metrics,
                }
                print(f'  {key}: cohesion={metrics["cohesion"]:.4f}, '
                      f'eff_rank={metrics["effective_rank"]:.2f}')

    # All four
    merged_fps = np.concatenate([domain_fingerprints[d] for d in domains], axis=0)
    merged_centroid = merged_fps.mean(axis=0)
    metrics = compute_all_metrics(merged_fps, merged_centroid, [])
    results['all'] = {
        'type': 'all',
        'domains': domains,
        'specialist_metrics': metrics,
    }
    print(f'  all: cohesion={metrics["cohesion"]:.4f}, '
          f'eff_rank={metrics["effective_rank"]:.2f}')

    return results


# ============================================================
# Part D: Correlation with retrieval accuracy + threshold
# ============================================================

def run_part_d(part_a_results, retrieval_path):
    """Correlate metrics with observed retrieval accuracy."""
    print('\n' + '=' * 60)
    print('PART D: Metric-Retrieval Correlation & Threshold')
    print('=' * 60)

    # Load retrieval results
    with open(retrieval_path) as f:
        retrieval = json.load(f)

    retrieval_acc = {}
    for d, stats in retrieval['summary']['by_domain'].items():
        retrieval_acc[d] = stats['top1_precision']

    print('\n  Domain retrieval accuracy:')
    for d, acc in sorted(retrieval_acc.items()):
        print(f'    {d}: {acc:.1%}')

    # Build correlation table
    domains = sorted(set(part_a_results.keys()) & set(retrieval_acc.keys()))
    metrics_names = ['cohesion', 'effective_rank', 'intra_variance',
                     'centroid_discriminability']

    print('\n  Correlation of metrics with retrieval accuracy:')
    correlations = {}
    for metric_name in metrics_names:
        # Use generalist metrics (since retrieval is a generalist-side operation)
        metric_vals = [part_a_results[d]['generalist_metrics'][metric_name]
                       for d in domains]
        acc_vals = [retrieval_acc[d] for d in domains]

        if len(metric_vals) < 3:
            print(f'    {metric_name}: too few domains for correlation')
            continue

        # Pearson correlation
        from scipy import stats as scipy_stats
        r, p = scipy_stats.pearsonr(metric_vals, acc_vals)
        correlations[metric_name] = {'pearson_r': float(r), 'p_value': float(p)}
        print(f'    {metric_name}: r={r:.4f}, p={p:.4f}')

        # Print per-domain values
        for d, m, a in zip(domains, metric_vals, acc_vals):
            print(f'      {d:10s}: metric={m:.4f}, retrieval={a:.1%}')

    # Identify best predictor
    if correlations:
        # For cohesion: higher = better retrieval (positive correlation)
        # For effective_rank: lower = better (negative correlation expected)
        # For centroid_discriminability: lower = better (negative correlation expected)
        best = max(correlations.items(), key=lambda x: abs(x[1]['pearson_r']))
        print(f'\n  Best predictor: {best[0]} (|r|={abs(best[1]["pearson_r"]):.4f})')

    return {
        'retrieval_accuracy': retrieval_acc,
        'correlations': correlations,
        'per_domain': {d: {
            'retrieval_top1': retrieval_acc.get(d, None),
            'generalist_metrics': part_a_results[d]['generalist_metrics'],
            'specialist_metrics': part_a_results[d]['specialist_metrics'],
        } for d in domains},
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 3: Cohesion Analysis')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--generalist-ckpt', type=str, required=True)
    parser.add_argument('--specialist-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--retrieval-json', type=str, default=None,
                        help='Path to pass2_retrieval.json (auto-detected if not set)')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect retrieval JSON
    if args.retrieval_json is None:
        # Look in sibling directory
        spec_parent = Path(args.specialist_dir).parent
        candidate = spec_parent / 'pass2_retrieval.json'
        if candidate.exists():
            args.retrieval_json = str(candidate)
        else:
            print(f'WARNING: No retrieval JSON found at {candidate}')

    # Configure model
    if args.model == 'gpt2':
        _set_model_cls(GPT2LMHeadModel)
        model_cfg = AutoConfig.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        _set_model_cls(AutoModelForCausalLM)
        model_cfg = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize probes
    print('Tokenizing domain probes...')
    probe_set = tokenize_domain_probes(tokenizer)
    print(f'  {len(probe_set["texts"])} probes across {len(DOMAINS)} domains')

    # Part A: Domain-level metrics
    part_a, domain_fps, domain_centroids, gen_fps, gen_centroids = run_part_a(
        args.specialist_dir, args.generalist_ckpt, model_cfg, probe_set, args.device)

    with open(output_dir / 'part_a_domain_metrics.json', 'w') as f:
        json.dump(part_a, f, indent=2, default=float)
    print(f'\n  Saved: {output_dir}/part_a_domain_metrics.json')

    # Part B: Medical subcategories
    part_b = run_part_b(domain_fps, domain_centroids, gen_fps, gen_centroids)

    with open(output_dir / 'part_b_medical_subcategories.json', 'w') as f:
        json.dump(part_b, f, indent=2, default=float)
    print(f'  Saved: {output_dir}/part_b_medical_subcategories.json')

    # Part C: Synthetic merges
    part_c = run_part_c(domain_fps, domain_centroids, gen_fps, gen_centroids)

    with open(output_dir / 'part_c_granularity_sweep.json', 'w') as f:
        json.dump(part_c, f, indent=2, default=float)
    print(f'  Saved: {output_dir}/part_c_granularity_sweep.json')

    # Part D: Correlation with retrieval
    if args.retrieval_json:
        part_d = run_part_d(part_a, args.retrieval_json)
        with open(output_dir / 'part_d_correlation.json', 'w') as f:
            json.dump(part_d, f, indent=2, default=float)
        print(f'  Saved: {output_dir}/part_d_correlation.json')
    else:
        print('\n  Skipping Part D (no retrieval JSON)')

    # Save all fingerprints for further analysis
    np.savez(output_dir / 'fingerprints.npz',
             **{f'specialist_{d}': domain_fps[d] for d in domain_fps},
             **{f'generalist_{d}': gen_fps[d] for d in gen_fps})
    print(f'  Saved: {output_dir}/fingerprints.npz')

    print('\n' + '=' * 60)
    print('COHESION ANALYSIS COMPLETE')
    print('=' * 60)


if __name__ == '__main__':
    main()
