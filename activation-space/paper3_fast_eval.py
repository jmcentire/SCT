#!/usr/bin/env python3
"""Fast evaluation for 7B+ models.

Keeps the model on GPU and only patches composed layers (last 4) instead of
reloading the entire state dict for every probe. 100x+ faster for large models.

Usage:
    python paper3_fast_eval.py --model Qwen/Qwen2.5-7B --device cuda --use-amp
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from paper3_refined import (
    gram_schmidt, decompose_query, compose_from_decomposition,
    task_arithmetic_compose, partial_fallback_blend, compute_val_loss,
    _set_model_cls, _get_model_cls,
)
from paper3_constellation import tokenize_domain_probes, DOMAINS
from src.fingerprint.capture import capture_at_checkpoint

try:
    from paper3_extended_probes import EXTENDED_CROSS_DOMAIN_PROBES
    _USE_EXTENDED_PROBES = True
except ImportError:
    _USE_EXTENDED_PROBES = False
    from paper3_constellation import CROSS_DOMAIN_PROBES


def get_compose_layer_keys(state_dict, n_compose=4):
    """Get the set of keys belonging to composed layers (last n)."""
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


def fast_load_partial(model, full_sd, keys_to_update, device):
    """Update only specific keys in the model, in-place on GPU."""
    with torch.no_grad():
        sd = model.state_dict()
        for k in keys_to_update:
            if k in full_sd:
                sd[k].copy_(full_sd[k].float().to(device))
        # No need to call load_state_dict — we modified tensors in-place


def fast_compose_inplace(model, gen_sd, spec_sds, weights, compose_keys, device):
    """Compose specialist parameters for compose_keys only, update model in-place."""
    spec_ids = [sid for sid in weights if sid in spec_sds and weights[sid] > 1e-6]
    if not spec_ids:
        fast_load_partial(model, gen_sd, compose_keys, device)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-single', action='store_true',
                        help='Skip single-domain eval')
    args = parser.parse_args()

    model_name_safe = args.model.replace('/', '_')
    results_dir = Path(f'results/paper3_refined/{model_name_safe}/seed{args.seed}')
    device = torch.device(args.device)

    from transformers import AutoModelForCausalLM, GPT2LMHeadModel
    model_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 'gpt2' in args.model.lower():
        _set_model_cls(GPT2LMHeadModel)
    else:
        _set_model_cls(AutoModelForCausalLM)

    # Load state dicts (half precision on CPU)
    gen_path = results_dir / 'step_02000.pt'
    print(f'Loading generalist from {gen_path}')
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
    gc.collect()

    # Get compose layer keys
    compose_keys, compose_layer_nums = get_compose_layer_keys(gen_sd)
    print(f'Compose layers: {sorted(compose_layer_nums)} ({len(compose_keys)} keys)')

    # Pre-compute TA state dicts for compose keys ONLY (saves memory)
    import sys
    print('Pre-computing task arithmetic compositions (compose keys only)...')
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
        print(f'  alpha={alpha} done ({len(ta_partial)} keys)')
        sys.stdout.flush()
    gc.collect()
    print(f'  TA pre-computation complete')

    # Load orthogonalization
    with open(results_dir / 'orthogonalization.json') as f:
        ortho_data = json.load(f)
    domain_order = ortho_data['domain_order']

    with open(results_dir / 'generalist_index.json') as f:
        index_data = json.load(f)
    # Index may be nested under 'domain_centroids' or flat
    centroids_dict = index_data.get('domain_centroids', index_data)
    centroid_vectors = [np.array(centroids_dict[d]) for d in domain_order]
    centroid_matrix = np.stack(centroid_vectors)
    ortho_basis, _ = gram_schmidt(centroid_matrix)

    # Create model on GPU with generalist weights
    print('Loading model to GPU...')
    model = _get_model_cls()(model_cfg)
    model.load_state_dict({k: v.float() for k, v in gen_sd.items()})
    model.to(device)
    model.eval()
    print(f'  Model on GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB')

    # === SINGLE-DOMAIN ===
    if not args.skip_single:
        print('\n' + '='*60)
        print('SINGLE-DOMAIN EVALUATION')
        print('='*60)

        probe_set = tokenize_domain_probes(tokenizer)
        NUM_PROBES = probe_set['input_ids'].shape[0]
        domain_labels = probe_set['domain_labels']

        # Generalist fingerprints
        print('Computing generalist fingerprints...')
        gen_fp = capture_at_checkpoint(model, probe_set, ['final_hidden'],
                                        device=str(device), batch_size=32)
        gen_activations = gen_fp['final_hidden']

        # Generalist baseline losses (model already has gen weights)
        print('Computing generalist baseline losses...')
        generalist_losses = []
        for i in range(NUM_PROBES):
            val_batch = {
                'input_ids': probe_set['input_ids'][i:i+1].to(device),
                'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
            }
            generalist_losses.append(compute_val_loss(model, val_batch))

        print(f'Evaluating {NUM_PROBES} single-domain probes...')
        single_results = []
        decomposition_log = []
        t0 = time.time()

        for i in range(NUM_PROBES):
            true_domain = domain_labels[i]
            query_fp = gen_activations[i]
            decomp = decompose_query(query_fp, ortho_basis, domain_order)

            # Compose — only patch compose layers
            fast_compose_inplace(model, gen_sd, spec_sds, decomp['weights'],
                                  compose_keys, device)
            val_batch = {
                'input_ids': probe_set['input_ids'][i:i+1].to(device),
                'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
            }
            composed_loss = compute_val_loss(model, val_batch)

            # Partial fallback
            gen_loss = generalist_losses[i]
            if composed_loss > gen_loss:
                # Blend: interpolate compose keys between gen and composed
                blend_ratio = 0.1
                with torch.no_grad():
                    sd = model.state_dict()
                    for k in compose_keys:
                        gen_val = gen_sd[k].float().to(device)
                        sd[k].mul_(1 - blend_ratio).add_(gen_val, alpha=blend_ratio)
                blended_loss = compute_val_loss(model, val_batch)
            else:
                blended_loss = composed_loss
                blend_ratio = 0.0

            # TA baseline — only patch compose keys
            fast_load_partial(model, gen_sd, compose_keys, device)
            ta_losses_dict = {}
            for ta_alpha in [0.3, 0.5, 1.0]:
                fast_load_partial(model, ta_compose_sds[ta_alpha], compose_keys, device)
                ta_losses_dict[ta_alpha] = compute_val_loss(model, val_batch)
            best_ta_alpha = min(ta_losses_dict, key=ta_losses_dict.get)
            best_ta_loss = ta_losses_dict[best_ta_alpha]

            # Oracle
            oracle_loss = float('inf')
            if true_domain in spec_sds and true_domain != 'general':
                fast_load_partial(model, spec_sds[true_domain], compose_keys, device)
                oracle_loss = compute_val_loss(model, val_batch)

            # Restore generalist
            fast_load_partial(model, gen_sd, compose_keys, device)

            single_results.append({
                'probe_idx': i,
                'true_domain': true_domain,
                'decomposition_weights': decomp['weights'],
                'residual_norm': decomp['residual_norm'],
                'sparsity_gini': decomp['sparsity_gini'],
                'generalist_loss': gen_loss,
                'composed_loss': composed_loss,
                'blended_loss': blended_loss,
                'blend_ratio': blend_ratio,
                'best_task_arithmetic_loss': best_ta_loss,
                'oracle_loss': oracle_loss if oracle_loss < float('inf') else None,
                'composed_beats_generalist': composed_loss < gen_loss,
                'blended_beats_generalist': blended_loss < gen_loss,
                'composed_beats_task_arithmetic': composed_loss < best_ta_loss,
                'blended_beats_task_arithmetic': blended_loss < best_ta_loss,
            })

            decomposition_log.append({
                'probe_idx': i,
                'true_domain': true_domain,
                'coefficients': decomp['coefficients'],
                'weights': decomp['weights'],
                'sparsity_gini': decomp['sparsity_gini'],
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (NUM_PROBES - i - 1) / rate if rate > 0 else 0
                wins = sum(r['composed_beats_generalist'] for r in single_results)
                wins_ta = sum(r['composed_beats_task_arithmetic'] for r in single_results)
                print(f'  {i+1}/{NUM_PROBES} ({elapsed:.0f}s, ETA {eta:.0f}s) '
                      f'vs_gen={wins}/{i+1} vs_ta={wins_ta}/{i+1}')

        single_summary = {
            'n_probes': len(single_results),
            'win_rate_composed_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in single_results])),
            'win_rate_blended_vs_generalist': float(np.mean([r['blended_beats_generalist'] for r in single_results])),
            'win_rate_composed_vs_task_arithmetic': float(np.mean([r['composed_beats_task_arithmetic'] for r in single_results])),
            'win_rate_blended_vs_task_arithmetic': float(np.mean([r['blended_beats_task_arithmetic'] for r in single_results])),
            'mean_composed_loss': float(np.mean([r['composed_loss'] for r in single_results])),
            'mean_blended_loss': float(np.mean([r['blended_loss'] for r in single_results])),
            'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in single_results])),
            'mean_sparsity_gini': float(np.mean([r['sparsity_gini'] for r in single_results])),
            'mean_residual_norm': float(np.mean([r['residual_norm'] for r in single_results])),
            'mean_blend_ratio': float(np.mean([r['blend_ratio'] for r in single_results])),
        }

        # Per-domain stats
        for domain in ['medical', 'legal', 'code', 'science']:
            domain_results = [r for r in single_results if r['true_domain'] == domain]
            if domain_results:
                single_summary[f'{domain}_win_rate_vs_gen'] = float(
                    np.mean([r['composed_beats_generalist'] for r in domain_results]))
                single_summary[f'{domain}_mean_sparsity'] = float(
                    np.mean([r['sparsity_gini'] for r in domain_results]))

        print(f'\n  Single-domain results:')
        print(f'    vs generalist: {single_summary["win_rate_composed_vs_generalist"]:.1%}')
        print(f'    vs task arith: {single_summary["win_rate_composed_vs_task_arithmetic"]:.1%}')
        print(f'    mean sparsity: {single_summary["mean_sparsity_gini"]:.3f}')
    else:
        single_results = []
        single_summary = {}
        decomposition_log = []

    # === CROSS-DOMAIN ===
    print('\n' + '='*60)
    print('CROSS-DOMAIN EVALUATION')
    print('='*60)

    if _USE_EXTENDED_PROBES:
        probe_list = EXTENDED_CROSS_DOMAIN_PROBES
    else:
        probe_list = CROSS_DOMAIN_PROBES
    print(f'Using {"extended" if _USE_EXTENDED_PROBES else "built-in"} probes: {len(probe_list)}')

    texts = [p[0] for p in probe_list]
    encoded = tokenizer(texts, padding='max_length', truncation=True,
                         max_length=128, return_tensors='pt')
    cross_probe_set = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
    }

    # Ensure generalist weights
    fast_load_partial(model, gen_sd, compose_keys, device)

    # Generalist fingerprints
    print('Computing generalist fingerprints...')
    cross_fp = capture_at_checkpoint(model, cross_probe_set, ['final_hidden'],
                                      device=str(device), batch_size=32)
    cross_activations = cross_fp['final_hidden']

    # Generalist baseline losses
    print('Computing generalist baseline losses...')
    cross_gen_losses = []
    for i in range(len(texts)):
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        cross_gen_losses.append(compute_val_loss(model, val_batch))

    print(f'Evaluating {len(probe_list)} cross-domain probes...')
    cross_results = []
    t0 = time.time()

    for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
        query_fp = cross_activations[i]
        decomp = decompose_query(query_fp, ortho_basis, domain_order)

        fast_compose_inplace(model, gen_sd, spec_sds, decomp['weights'],
                              compose_keys, device)
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(model, val_batch)

        gen_loss = cross_gen_losses[i]
        if composed_loss > gen_loss:
            blend_ratio = 0.1
            with torch.no_grad():
                sd = model.state_dict()
                for k in compose_keys:
                    gen_val = gen_sd[k].float().to(device)
                    sd[k].mul_(1 - blend_ratio).add_(gen_val, alpha=blend_ratio)
            blended_loss = compute_val_loss(model, val_batch)
        else:
            blended_loss = composed_loss
            blend_ratio = 0.0

        # TA baseline
        fast_load_partial(model, gen_sd, compose_keys, device)
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            fast_load_partial(model, ta_compose_sds[ta_alpha], compose_keys, device)
            ta_losses[ta_alpha] = compute_val_loss(model, val_batch)
        best_ta_alpha = min(ta_losses, key=ta_losses.get)
        best_ta_loss = ta_losses[best_ta_alpha]

        # Best single specialist
        specialist_losses = {}
        for sid in [dom_a, dom_b]:
            if sid in spec_sds:
                fast_load_partial(model, spec_sds[sid], compose_keys, device)
                specialist_losses[sid] = compute_val_loss(model, val_batch)
        best_single = min(specialist_losses.values()) if specialist_losses else float('inf')

        # Restore generalist
        fast_load_partial(model, gen_sd, compose_keys, device)

        cross_results.append({
            'probe_idx': i,
            'text_preview': text[:80],
            'expected_domains': [dom_a, dom_b],
            'decomposition_weights': decomp['weights'],
            'residual_norm': decomp['residual_norm'],
            'sparsity_gini': decomp['sparsity_gini'],
            'generalist_loss': gen_loss,
            'composed_loss': composed_loss,
            'blended_loss': blended_loss,
            'blend_ratio': blend_ratio,
            'best_task_arithmetic_loss': best_ta_loss,
            'best_single_specialist_loss': best_single if best_single < float('inf') else None,
            'composed_beats_generalist': composed_loss < gen_loss,
            'blended_beats_generalist': blended_loss < gen_loss,
            'composed_beats_best_single': composed_loss < best_single,
            'composed_beats_task_arithmetic': composed_loss < best_ta_loss,
            'blended_beats_task_arithmetic': blended_loss < best_ta_loss,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(probe_list) - i - 1) / rate if rate > 0 else 0
            wins_gen = sum(r['composed_beats_generalist'] for r in cross_results)
            wins_ta = sum(r['composed_beats_task_arithmetic'] for r in cross_results)
            print(f'  {i+1}/{len(probe_list)} ({elapsed:.0f}s, ETA {eta:.0f}s) '
                  f'vs_gen={wins_gen}/{i+1} vs_ta={wins_ta}/{i+1}')

    cross_summary = {
        'n_probes': len(cross_results),
        'win_rate_composed_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in cross_results])),
        'win_rate_blended_vs_generalist': float(np.mean([r['blended_beats_generalist'] for r in cross_results])),
        'win_rate_composed_vs_best_single': float(np.mean([r['composed_beats_best_single'] for r in cross_results])),
        'win_rate_composed_vs_task_arithmetic': float(np.mean([r['composed_beats_task_arithmetic'] for r in cross_results])),
        'win_rate_blended_vs_task_arithmetic': float(np.mean([r['blended_beats_task_arithmetic'] for r in cross_results])),
        'mean_composed_loss': float(np.mean([r['composed_loss'] for r in cross_results])),
        'mean_blended_loss': float(np.mean([r['blended_loss'] for r in cross_results])),
        'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in cross_results])),
        'mean_sparsity_gini': float(np.mean([r['sparsity_gini'] for r in cross_results])),
    }

    print(f'\n  Cross-domain results ({len(cross_results)} probes):')
    print(f'    vs generalist:      {cross_summary["win_rate_composed_vs_generalist"]:.1%}')
    print(f'    vs task arithmetic: {cross_summary["win_rate_composed_vs_task_arithmetic"]:.1%}')
    print(f'    vs best single:    {cross_summary["win_rate_composed_vs_best_single"]:.1%}')
    print(f'    mean sparsity:     {cross_summary["mean_sparsity_gini"]:.3f}')

    # === SAVE ===
    output = {
        'model': args.model,
        'seed': args.seed,
        'domain_order': domain_order,
        'orthogonalization': ortho_data,
        'single_domain': single_summary,
        'cross_domain': cross_summary,
    }

    out_path = results_dir / 'paper3_refined_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f'\nResults saved to {out_path}')

    if single_results:
        with open(results_dir / 'single_domain_results.json', 'w') as f:
            json.dump(single_results, f, indent=2, default=float)

    with open(results_dir / 'cross_domain_results.json', 'w') as f:
        json.dump(cross_results, f, indent=2, default=float)

    if decomposition_log:
        with open(results_dir / 'decomposition_log.json', 'w') as f:
            json.dump(decomposition_log, f, indent=2, default=float)

    print('DONE')


if __name__ == '__main__':
    main()
