#!/usr/bin/env python3
"""Stochastic resonance experiment for constellation composition.

Hypothesis: At high collinearity, the orthogonal components after Gram-Schmidt
have very small norms — the signal-to-noise ratio for decomposition is low.
Adding calibrated noise to the centroid vectors before orthogonalization may
boost weak discriminative signals above the detection threshold (stochastic
resonance).

Two approaches:
  1. Single-shot: add noise to centroids, re-orthogonalize, decompose, evaluate
  2. Ensemble: N noise realizations, average the composition weights, evaluate

Usage:
    python paper3_stochastic_resonance.py --model Qwen/Qwen2.5-3B --device cuda --use-amp
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from paper3_refined import (
    gram_schmidt, decompose_query, compute_val_loss,
    _set_model_cls, _get_model_cls, get_batch_sizes,
)
from src.fingerprint.capture import capture_at_checkpoint

try:
    from paper3_extended_probes import EXTENDED_CROSS_DOMAIN_PROBES
    _USE_EXTENDED_PROBES = True
except ImportError:
    _USE_EXTENDED_PROBES = False
    from paper3_constellation import CROSS_DOMAIN_PROBES


def get_compose_layer_keys(state_dict, n_compose=4):
    """Get keys belonging to composed layers (last n)."""
    all_layer_nums = set()
    for name in state_dict:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p in ('h', 'layers') and i + 1 < len(parts):
                try:
                    all_layer_nums.add(int(parts[i + 1]))
                except ValueError:
                    pass
    if not all_layer_nums:
        return set(), set()
    max_layer = max(all_layer_nums)
    compose_layer_nums = set(range(max(0, max_layer - n_compose + 1), max_layer + 1))
    compose_keys = set()
    for name in state_dict:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p in ('h', 'layers') and i + 1 < len(parts):
                try:
                    if int(parts[i + 1]) in compose_layer_nums:
                        compose_keys.add(name)
                except ValueError:
                    pass
    return compose_keys, compose_layer_nums


def fast_compose_inplace(model, gen_sd, spec_sds, weights, compose_keys, device):
    """Compose specialist parameters for compose_keys only, in-place."""
    spec_ids = [sid for sid in weights if sid in spec_sds and weights[sid] > 1e-6]
    if not spec_ids:
        with torch.no_grad():
            sd = model.state_dict()
            for k in compose_keys:
                if k in gen_sd:
                    sd[k].copy_(gen_sd[k].float().to(device))
        return
    total_w = sum(weights[sid] for sid in spec_ids)
    with torch.no_grad():
        sd = model.state_dict()
        for name in compose_keys:
            if not all(name in spec_sds[sid] for sid in spec_ids):
                sd[name].copy_(gen_sd[name].float().to(device))
                continue
            composed = torch.zeros_like(sd[name])
            for sid in spec_ids:
                w = weights[sid] / total_w
                composed.add_(spec_sds[sid][name].float().to(device), alpha=w)
            sd[name].copy_(composed)


def fast_load_partial(model, full_sd, keys_to_update, device):
    """Update only specific keys in the model, in-place."""
    with torch.no_grad():
        sd = model.state_dict()
        for k in keys_to_update:
            if k in full_sd:
                sd[k].copy_(full_sd[k].float().to(device))


def evaluate_with_basis(model, ortho_basis, domain_order, gen_sd, spec_sds,
                        ta_compose_sds, cross_activations, cross_probe_set,
                        cross_gen_losses, compose_keys, device, probe_list):
    """Evaluate cross-domain probes with a given orthogonal basis."""
    wins_gen = 0
    wins_ta = 0
    n = len(probe_list)

    for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
        query_fp = cross_activations[i]
        decomp = decompose_query(query_fp, ortho_basis, domain_order)

        # Compose
        fast_compose_inplace(model, gen_sd, spec_sds, decomp['weights'],
                             compose_keys, device)
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(model, val_batch)

        # TA baseline
        fast_load_partial(model, gen_sd, compose_keys, device)
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            fast_load_partial(model, ta_compose_sds[ta_alpha], compose_keys, device)
            ta_losses[ta_alpha] = compute_val_loss(model, val_batch)
        best_ta_loss = min(ta_losses.values())

        gen_loss = cross_gen_losses[i]

        if composed_loss < gen_loss:
            wins_gen += 1
        if composed_loss < best_ta_loss:
            wins_ta += 1

        # Restore generalist
        fast_load_partial(model, gen_sd, compose_keys, device)

    return {
        'win_rate_vs_gen': wins_gen / n,
        'win_rate_vs_ta': wins_ta / n,
        'wins_gen': wins_gen,
        'wins_ta': wins_ta,
        'n': n,
    }


def evaluate_ensemble(model, centroid_matrix, domain_order, gen_sd, spec_sds,
                      ta_compose_sds, cross_activations, cross_probe_set,
                      cross_gen_losses, compose_keys, device, probe_list,
                      noise_sigma, n_ensemble=5, rng=None):
    """Evaluate using ensemble of noisy orthogonalizations — average weights."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(probe_list)
    wins_gen = 0
    wins_ta = 0

    for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
        query_fp = cross_activations[i]

        # Average decomposition weights across noise realizations
        avg_weights = {d: 0.0 for d in domain_order}
        for _ in range(n_ensemble):
            noise = rng.normal(0, noise_sigma, centroid_matrix.shape)
            noisy_centroids = centroid_matrix + noise
            noisy_basis, _ = gram_schmidt(noisy_centroids)
            decomp = decompose_query(query_fp, noisy_basis, domain_order)
            for d in domain_order:
                avg_weights[d] += decomp['weights'][d] / n_ensemble

        # Compose with averaged weights
        fast_compose_inplace(model, gen_sd, spec_sds, avg_weights,
                             compose_keys, device)
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(model, val_batch)

        # TA baseline
        fast_load_partial(model, gen_sd, compose_keys, device)
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            fast_load_partial(model, ta_compose_sds[ta_alpha], compose_keys, device)
            ta_losses[ta_alpha] = compute_val_loss(model, val_batch)
        best_ta_loss = min(ta_losses.values())

        gen_loss = cross_gen_losses[i]
        if composed_loss < gen_loss:
            wins_gen += 1
        if composed_loss < best_ta_loss:
            wins_ta += 1

        fast_load_partial(model, gen_sd, compose_keys, device)

    return {
        'win_rate_vs_gen': wins_gen / n,
        'win_rate_vs_ta': wins_ta / n,
        'wins_gen': wins_gen,
        'wins_ta': wins_ta,
        'n': n,
    }


def measure_orthogonal_norms(centroid_matrix):
    """Measure norms of Gram-Schmidt residuals (the orthogonal components).

    These norms quantify the discriminative signal per domain. At high
    collinearity, these shrink toward zero — the signal-to-noise problem
    that motivates stochastic resonance.
    """
    n, d = centroid_matrix.shape
    residual_norms = []
    vectors = centroid_matrix.astype(np.float64)
    orthogonal = np.zeros_like(vectors)

    for i in range(n):
        v = vectors[i].copy()
        for j in range(i):
            v = v - np.dot(v, orthogonal[j]) * orthogonal[j]
        norm = np.linalg.norm(v)
        residual_norms.append(float(norm))
        if norm > 1e-10:
            orthogonal[i] = v / norm
        else:
            orthogonal[i] = v

    return residual_norms


def jensen_gap_test(centroid_matrix, domain_order, noise_sigma, n_samples=20, rng=None):
    """Measure the Jensen gap: E[norms(GS(x+noise))] vs norms(GS(x)).

    If mean noisy norms > clean norms for later components, the nonlinearity
    of Gram-Schmidt creates a beneficial bias — the mechanism behind SR.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    clean_norms = measure_orthogonal_norms(centroid_matrix)

    noisy_norm_samples = []
    for _ in range(n_samples):
        noise = rng.normal(0, noise_sigma, centroid_matrix.shape)
        noisy = centroid_matrix + noise
        noisy_norm_samples.append(measure_orthogonal_norms(noisy))

    mean_noisy_norms = np.mean(noisy_norm_samples, axis=0).tolist()
    std_noisy_norms = np.std(noisy_norm_samples, axis=0).tolist()

    gap = [mn - cn for mn, cn in zip(mean_noisy_norms, clean_norms)]

    return {
        'domain_order': domain_order,
        'clean_norms': clean_norms,
        'mean_noisy_norms': mean_noisy_norms,
        'std_noisy_norms': std_noisy_norms,
        'jensen_gap': gap,
        'noise_sigma': float(noise_sigma),
        'n_samples': n_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-probes', type=int, default=0,
                        help='Limit probes for faster testing (0=all)')
    args = parser.parse_args()

    model_name_safe = args.model.replace('/', '_')
    results_dir = Path(f'results/paper3_refined/{model_name_safe}/seed{args.seed}')
    device = torch.device(args.device)

    if args.model == 'gpt2':
        from transformers import GPT2LMHeadModel
        _set_model_cls(GPT2LMHeadModel)
        model_cfg = AutoConfig.from_pretrained('gpt2')
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
    else:
        _set_model_cls(AutoModelForCausalLM)
        model_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, capture_bs = get_batch_sizes(model_cfg)

    # Load state dicts
    gen_path = results_dir / 'step_02000.pt'
    print(f'Loading generalist from {gen_path}')
    sys.stdout.flush()
    gen_ckpt = torch.load(gen_path, weights_only=False, map_location='cpu')
    gen_sd = {k: v.half() for k, v in gen_ckpt['state_dict'].items()}
    del gen_ckpt; gc.collect()

    spec_sds = {}
    for domain in ['medical', 'legal', 'code', 'science']:
        spec_path = results_dir / 'specialists' / domain / 'specialist.pt'
        if spec_path.exists():
            ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
            spec_sds[domain] = {k: v.half() for k, v in ckpt['state_dict'].items()}
            del ckpt
            print(f'  Loaded {domain}')
            sys.stdout.flush()
    gc.collect()

    # Compose layer keys
    compose_keys, compose_layer_nums = get_compose_layer_keys(gen_sd)
    print(f'Compose layers: {sorted(compose_layer_nums)} ({len(compose_keys)} keys)')
    sys.stdout.flush()

    # Pre-compute TA compositions for compose keys
    print('Pre-computing TA compositions...')
    sys.stdout.flush()
    ta_compose_sds = {}
    for alpha in [0.3, 0.5, 1.0]:
        ta_partial = {}
        for k in compose_keys:
            if k in gen_sd:
                gen_val = gen_sd[k].float()
                task_sum = gen_val.clone()
                for sid, ssd in spec_sds.items():
                    if k in ssd:
                        task_sum = task_sum + alpha * (ssd[k].float() - gen_val)
                ta_partial[k] = task_sum.half()
                del gen_val, task_sum
        ta_compose_sds[alpha] = ta_partial
        print(f'  alpha={alpha} done')
        sys.stdout.flush()
    gc.collect()

    # Load centroids
    with open(results_dir / 'orthogonalization.json') as f:
        ortho_data = json.load(f)
    domain_order = ortho_data['domain_order']

    with open(results_dir / 'generalist_index.json') as f:
        index_data = json.load(f)
    centroids_dict = index_data.get('domain_centroids', index_data)
    centroid_vectors = [np.array(centroids_dict[d]) for d in domain_order]
    centroid_matrix = np.stack(centroid_vectors)

    # Baseline orthogonal basis (no noise)
    baseline_basis, _ = gram_schmidt(centroid_matrix)

    # Compute centroid norms for noise calibration
    norms = np.linalg.norm(centroid_matrix, axis=1)
    mean_norm = np.mean(norms)
    collinearity = ortho_data['mean_off_diagonal_similarity']
    print(f'Centroid mean norm: {mean_norm:.4f}')
    print(f'Centroid collinearity: {collinearity:.4f}')
    sys.stdout.flush()

    # === COMMUNICATIVE VARIANCE DIAGNOSTICS ===
    # Measure orthogonal residual norms (signal strength per domain)
    residual_norms = measure_orthogonal_norms(centroid_matrix)
    print(f'\nOrthogonal residual norms (discriminative signal per domain):')
    for d, rn in zip(domain_order, residual_norms):
        print(f'  {d}: {rn:.4f}  (ratio to centroid: {rn/mean_norm:.4f})')
    print(f'  Mean residual/centroid ratio: {np.mean(residual_norms)/mean_norm:.4f}')
    sys.stdout.flush()

    # Jensen gap test at multiple noise levels
    print(f'\nJensen gap test (C2 nonlinearity check):')
    jensen_results = []
    for test_frac in [0.005, 0.02, 0.05]:
        test_sigma = test_frac * mean_norm
        jg = jensen_gap_test(centroid_matrix, domain_order, test_sigma,
                             n_samples=50, rng=np.random.default_rng(42))
        jensen_results.append(jg)
        gaps = jg['jensen_gap']
        print(f'  sigma={test_sigma:.4f} (frac={test_frac}): '
              f'gaps={[f"{g:+.4f}" for g in gaps]}')
    sys.stdout.flush()

    # Condition assessment
    print(f'\n--- Communicative Variance Condition Check ---')
    print(f'  C1 (Suboptimal): collinearity={collinearity:.4f} '
          f'{"MET" if collinearity > 0.95 else "MARGINAL" if collinearity > 0.92 else "LIKELY NOT MET"}')
    print(f'  C2 (Nonlinear):  Gram-Schmidt is nonlinear — MET by construction')
    min_gap = min(jensen_results[-1]['jensen_gap'][1:])  # skip first (always ~0)
    print(f'  C2 (Jensen gap): min gap at sigma=0.05*norm: {min_gap:+.4f} '
          f'{"POSITIVE — noise helps" if min_gap > 0 else "NEGATIVE or ZERO"}')
    print(f'  C3 (Accessible):  Gaussian noise spans all directions — MET')
    print(f'  C4 (Weighting):   Empirical — will be tested in sweep')
    print(f'  C5 (Robustness):  Centroids are averages over {len(centroid_matrix[0])}-dim vectors — LIKELY MET')
    sys.stdout.flush()

    # Create model
    print('Loading model to GPU...')
    sys.stdout.flush()
    model = _get_model_cls()(model_cfg)
    model.load_state_dict({k: v.float() for k, v in gen_sd.items()})
    model.to(device)
    model.eval()
    print(f'  Model on GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB')
    sys.stdout.flush()

    # Prepare cross-domain probes
    if _USE_EXTENDED_PROBES:
        probe_list = EXTENDED_CROSS_DOMAIN_PROBES
    else:
        probe_list = CROSS_DOMAIN_PROBES
    if args.max_probes > 0:
        probe_list = probe_list[:args.max_probes]
    print(f'Using {len(probe_list)} cross-domain probes')
    sys.stdout.flush()

    texts = [p[0] for p in probe_list]
    encoded = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=128, return_tensors='pt')
    cross_probe_set = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
    }

    # Generalist fingerprints
    print('Computing generalist fingerprints...')
    sys.stdout.flush()
    cross_fp = capture_at_checkpoint(model, cross_probe_set, ['final_hidden'],
                                     device=str(device), batch_size=capture_bs)
    cross_activations = cross_fp['final_hidden']

    # Generalist baseline losses
    print('Computing generalist baseline losses...')
    sys.stdout.flush()
    cross_gen_losses = []
    for i in range(len(texts)):
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        cross_gen_losses.append(compute_val_loss(model, val_batch))

    # === BASELINE (no noise) ===
    print('\n' + '='*60)
    print('BASELINE (no noise)')
    print('='*60)
    sys.stdout.flush()
    t0 = time.time()
    baseline = evaluate_with_basis(
        model, baseline_basis, domain_order, gen_sd, spec_sds,
        ta_compose_sds, cross_activations, cross_probe_set,
        cross_gen_losses, compose_keys, device, probe_list)
    baseline_time = time.time() - t0
    print(f'  vs gen: {baseline["win_rate_vs_gen"]:.1%}  vs TA: {baseline["win_rate_vs_ta"]:.1%}  ({baseline_time:.0f}s)')
    sys.stdout.flush()

    # === NOISE SWEEP (single-shot) ===
    # Noise sigma as fraction of mean centroid norm
    noise_fractions = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    print('\n' + '='*60)
    print('SINGLE-SHOT NOISE SWEEP')
    print('='*60)
    sys.stdout.flush()

    single_shot_results = []
    rng = np.random.default_rng(42)
    for frac in noise_fractions:
        sigma = frac * mean_norm
        noise = rng.normal(0, sigma, centroid_matrix.shape)
        noisy_centroids = centroid_matrix + noise
        noisy_basis, _ = gram_schmidt(noisy_centroids)

        t0 = time.time()
        result = evaluate_with_basis(
            model, noisy_basis, domain_order, gen_sd, spec_sds,
            ta_compose_sds, cross_activations, cross_probe_set,
            cross_gen_losses, compose_keys, device, probe_list)
        elapsed = time.time() - t0
        result['noise_fraction'] = frac
        result['noise_sigma'] = sigma
        single_shot_results.append(result)
        delta_ta = result['win_rate_vs_ta'] - baseline['win_rate_vs_ta']
        print(f'  frac={frac:.3f} sigma={sigma:.4f}: '
              f'vs_gen={result["win_rate_vs_gen"]:.1%} '
              f'vs_ta={result["win_rate_vs_ta"]:.1%} '
              f'(delta={delta_ta:+.1%}) ({elapsed:.0f}s)')
        sys.stdout.flush()

    # === ENSEMBLE NOISE (average over N realizations) ===
    print('\n' + '='*60)
    print('ENSEMBLE NOISE (5 realizations, averaged weights)')
    print('='*60)
    sys.stdout.flush()

    # Use the best single-shot fraction and nearby
    best_single = max(single_shot_results, key=lambda r: r['win_rate_vs_ta'])
    test_fracs = [best_single['noise_fraction'] * 0.5,
                  best_single['noise_fraction'],
                  best_single['noise_fraction'] * 2.0]
    # Deduplicate and bound
    test_fracs = sorted(set(max(0.001, min(0.2, f)) for f in test_fracs))

    ensemble_results = []
    for frac in test_fracs:
        sigma = frac * mean_norm
        rng_ens = np.random.default_rng(42)
        t0 = time.time()
        result = evaluate_ensemble(
            model, centroid_matrix, domain_order, gen_sd, spec_sds,
            ta_compose_sds, cross_activations, cross_probe_set,
            cross_gen_losses, compose_keys, device, probe_list,
            noise_sigma=sigma, n_ensemble=5, rng=rng_ens)
        elapsed = time.time() - t0
        result['noise_fraction'] = frac
        result['noise_sigma'] = sigma
        result['n_ensemble'] = 5
        ensemble_results.append(result)
        delta_ta = result['win_rate_vs_ta'] - baseline['win_rate_vs_ta']
        print(f'  frac={frac:.3f} sigma={sigma:.4f} n=5: '
              f'vs_gen={result["win_rate_vs_gen"]:.1%} '
              f'vs_ta={result["win_rate_vs_ta"]:.1%} '
              f'(delta={delta_ta:+.1%}) ({elapsed:.0f}s)')
        sys.stdout.flush()

    # === SUMMARY ===
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Baseline:    vs_gen={baseline["win_rate_vs_gen"]:.1%}  vs_ta={baseline["win_rate_vs_ta"]:.1%}')
    best_single = max(single_shot_results, key=lambda r: r['win_rate_vs_ta'])
    print(f'Best single: vs_gen={best_single["win_rate_vs_gen"]:.1%}  '
          f'vs_ta={best_single["win_rate_vs_ta"]:.1%}  '
          f'frac={best_single["noise_fraction"]:.3f}')
    if ensemble_results:
        best_ens = max(ensemble_results, key=lambda r: r['win_rate_vs_ta'])
        print(f'Best ensemb: vs_gen={best_ens["win_rate_vs_gen"]:.1%}  '
              f'vs_ta={best_ens["win_rate_vs_ta"]:.1%}  '
              f'frac={best_ens["noise_fraction"]:.3f}')
    sys.stdout.flush()

    # Save
    output = {
        'model': args.model,
        'seed': args.seed,
        'collinearity': collinearity,
        'centroid_mean_norm': float(mean_norm),
        'orthogonal_residual_norms': {d: rn for d, rn in zip(domain_order, residual_norms)},
        'residual_to_centroid_ratio': float(np.mean(residual_norms) / mean_norm),
        'jensen_gap_tests': jensen_results,
        'condition_check': {
            'C1_suboptimal': collinearity > 0.95,
            'C1_collinearity': collinearity,
            'C2_nonlinear': True,
            'C2_jensen_gap_positive': min(jensen_results[-1]['jensen_gap'][1:]) > 0,
            'C3_accessible': True,
            'C5_robust': True,
        },
        'baseline': baseline,
        'single_shot_sweep': single_shot_results,
        'ensemble_sweep': ensemble_results,
    }
    out_path = results_dir / 'stochastic_resonance_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f'\nSaved to {out_path}')
    print('DONE')


if __name__ == '__main__':
    main()
