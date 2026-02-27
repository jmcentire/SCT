#!/usr/bin/env python3
"""Standalone cross-domain evaluation with extended probes (150).

Run after paper3_refined.py to upgrade cross-domain from 30 to 150 probes.
Uses saved generalist checkpoint and specialist state dicts from the results dir.

Usage:
    python paper3_crossdomain_eval.py --model Qwen/Qwen2.5-7B --device cuda --use-amp
    python paper3_crossdomain_eval.py --model Qwen/Qwen2.5-3B --device cuda --use-amp
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from paper3_extended_probes import EXTENDED_CROSS_DOMAIN_PROBES
from paper3_refined import (
    gram_schmidt, decompose_query, compose_from_decomposition,
    task_arithmetic_compose, partial_fallback_blend, compute_val_loss,
    _set_model_cls, _get_model_cls,
)
from src.fingerprint.capture import capture_at_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    model_name_safe = args.model.replace('/', '_')
    results_dir = Path(f'results/paper3_refined/{model_name_safe}/seed{args.seed}')

    # Load config
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cfg = {
        'name': args.model,
        'd_model': config.hidden_size,
        'n_layers': config.num_hidden_layers,
    }
    _set_model_cls(args.model)

    device = torch.device(args.device)

    # Load generalist
    gen_path = results_dir / 'step_02000.pt'
    print(f'Loading generalist from {gen_path}')
    gen_ckpt = torch.load(gen_path, weights_only=False, map_location='cpu')
    gen_sd_half = {k: v.half() for k, v in gen_ckpt['state_dict'].items()}
    gen_sd_float = {k: v.float() for k, v in gen_sd_half.items()}
    del gen_ckpt
    gc.collect()

    # Load specialists
    spec_sds_float = {}
    for domain in ['medical', 'legal', 'code', 'science']:
        spec_path = results_dir / 'specialists' / domain / 'specialist.pt'
        if spec_path.exists():
            ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
            spec_sds_float[domain] = {k: v.float() for k, v in ckpt['state_dict'].items()}
            del ckpt
            print(f'  Loaded {domain} specialist')
        else:
            print(f'  WARNING: {domain} specialist not found')
    gc.collect()

    # Load orthogonalization results
    ortho_path = results_dir / 'orthogonalization.json'
    with open(ortho_path) as f:
        ortho_data = json.load(f)
    domain_order = ortho_data['domain_order']

    # Build orthogonal basis from generalist index
    index_path = results_dir / 'generalist_index.json'
    with open(index_path) as f:
        index_data = json.load(f)

    centroids_dict = index_data.get('domain_centroids', index_data)
    centroid_vectors = [torch.tensor(centroids_dict[d]) for d in domain_order]
    centroid_matrix = torch.stack(centroid_vectors)
    ortho_basis, _ = gram_schmidt(centroid_matrix)

    # Tokenize extended cross-domain probes
    probe_list = EXTENDED_CROSS_DOMAIN_PROBES
    print(f'\nUsing extended probes: {len(probe_list)} total')
    texts = [p[0] for p in probe_list]
    encoded = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=128, return_tensors='pt',
    )
    cross_probe_set = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
    }

    # Create model
    reuse_model = _get_model_cls()(model_cfg)

    def _load_sd(model, sd):
        model.load_state_dict({k: v.float() for k, v in sd.items()})

    _load_sd(reuse_model, gen_sd_float)
    reuse_model.to(device)

    # Generalist fingerprints
    print('Computing generalist fingerprints for cross-domain probes...')
    cross_fp = capture_at_checkpoint(reuse_model, cross_probe_set, ['final_hidden'],
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
        cross_gen_losses.append(compute_val_loss(reuse_model, val_batch))
    print(f'  Done. Mean generalist loss: {np.mean(cross_gen_losses):.4f}')

    # Evaluate cross-domain probes
    print(f'\nEvaluating {len(probe_list)} cross-domain probes...')
    cross_results = []
    t0 = time.time()

    for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
        query_fp = cross_activations[i]
        decomp = decompose_query(query_fp, ortho_basis, domain_order)

        composed_sd = compose_from_decomposition(
            gen_sd_float, spec_sds_float, decomp['weights'])
        _load_sd(reuse_model, composed_sd)
        val_batch = {
            'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(reuse_model, val_batch)

        gen_loss = cross_gen_losses[i]
        blended_sd, blend_ratio = partial_fallback_blend(
            gen_sd_float, composed_sd, gen_loss, composed_loss)
        if blend_ratio > 0.01:
            _load_sd(reuse_model, blended_sd)
            blended_loss = compute_val_loss(reuse_model, val_batch)
        else:
            blended_loss = composed_loss

        _load_sd(reuse_model, gen_sd_float)
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            ta_sd = task_arithmetic_compose(gen_sd_float, spec_sds_float, alpha=ta_alpha)
            _load_sd(reuse_model, ta_sd)
            ta_losses[ta_alpha] = compute_val_loss(reuse_model, val_batch)
        best_ta_alpha = min(ta_losses, key=ta_losses.get)
        best_ta_loss = ta_losses[best_ta_alpha]

        specialist_losses = {}
        for sid in [dom_a, dom_b]:
            if sid in spec_sds_float:
                _load_sd(reuse_model, spec_sds_float[sid])
                specialist_losses[sid] = compute_val_loss(reuse_model, val_batch)
        best_single = min(specialist_losses.values()) if specialist_losses else float('inf')

        _load_sd(reuse_model, gen_sd_float)

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
            'best_task_arithmetic_alpha': best_ta_alpha,
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

    # Summary
    summary = {
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

    print(f'\n=== CROSS-DOMAIN RESULTS (n={len(cross_results)}) ===')
    print(f'  vs generalist:       {summary["win_rate_composed_vs_generalist"]:.1%}')
    print(f'  vs task arithmetic:  {summary["win_rate_composed_vs_task_arithmetic"]:.1%}')
    print(f'  vs best single:     {summary["win_rate_composed_vs_best_single"]:.1%}')
    print(f'  mean sparsity:      {summary["mean_sparsity_gini"]:.3f}')

    # Save
    out_path = results_dir / 'cross_domain_extended_results.json'
    with open(out_path, 'w') as f:
        json.dump({'evaluations': cross_results, 'summary': summary},
                  f, indent=2, default=float)
    print(f'\nSaved to {out_path}')

    # Update main results file
    main_results_path = results_dir / 'paper3_refined_results.json'
    if main_results_path.exists():
        with open(main_results_path) as f:
            main_results = json.load(f)
        main_results['cross_domain'] = summary
        main_results['cross_domain_note'] = 'Updated with 150 extended probes'
        with open(main_results_path, 'w') as f:
            json.dump(main_results, f, indent=2, default=float)
        print(f'Updated {main_results_path}')


if __name__ == '__main__':
    main()
