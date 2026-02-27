#!/usr/bin/env python3
"""Paper 3 Refined: Generalist-Space Decomposition Composition.

Implements the original vision: generalist activation space as coordinate system,
specialist domain centroids as basis vectors, Fourier-like decomposition to derive
composition weights, with fact-checking fallback.

Key improvements over paper3_constellation.py:
  1. Generalist-space indexing: build index from generalist fingerprints, not specialist
  2. Gram-Schmidt orthogonalization of domain centroids before decomposition
  3. Decomposition-based weights: project query onto orthogonalized basis, coefficients
     ARE the composition weights (unifies retrieval + composition)
  4. Sparse decomposition logging (for Paper 5 PoH signal)
  5. Partial (blended) fallback by loss ratio, not binary switching
  6. Architecture invariance: save per-query decomposition coefficients for cross-model comparison

Usage:
  # Full pipeline: train specialists + all evaluation passes
  python paper3_refined.py --model Qwen/Qwen2.5-1.5B --device cuda --run-all

  # Train specialists only
  python paper3_refined.py --model Qwen/Qwen2.5-1.5B --device cuda --train-specialists

  # Evaluation only (specialists already trained)
  python paper3_refined.py --model Qwen/Qwen2.5-1.5B --device cuda --evaluate
"""
import os, sys, time, json, gc, argparse, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(__file__) or '.')

from src.fingerprint.capture import capture_at_checkpoint
from src.speculative.predictors import compute_val_loss
from src.constellation.compose import (
    task_arithmetic_compose, weight_entropy,
)

# Import domain probes and training infrastructure from paper3
from paper3_constellation import (
    DOMAINS, DOMAIN_PROBES, PROBES_PER_DOMAIN, NUM_PROBES,
    SPECIALIST_TRAIN_STEPS, SPECIALIST_BATCH_SIZE, SPECIALIST_LR,
    SPECIALIST_MAX_LENGTH, TOP_K_SPECIALISTS, COMPOSE_LAST_N_LAYERS,
    SimpleTokenDataset, get_lr, load_domain_data, tokenize_domain_probes,
    _set_model_cls, _get_model_cls,
)

# Extended cross-domain probes
try:
    from paper3_extended_probes import EXTENDED_CROSS_DOMAIN_PROBES
    _USE_EXTENDED_PROBES = True
except ImportError:
    _USE_EXTENDED_PROBES = False

# Also import the built-in cross-domain probes as fallback
from paper3_constellation import CROSS_DOMAIN_PROBES


# ============================================================
# Adaptive batch sizing for different model scales
# ============================================================

_TRAIN_BS = SPECIALIST_BATCH_SIZE
_CAPTURE_BS = 32


def get_batch_sizes(model_cfg):
    """Return (train_batch_size, capture_batch_size) based on model size.

    7B+ models need smaller batches to fit in 80GB VRAM with AdamW optimizer states.
    """
    n_params = getattr(model_cfg, 'num_parameters', None)
    if n_params is None:
        # Estimate from config: attention + FFN + embedding
        h = getattr(model_cfg, 'hidden_size', 768)
        n_layers = getattr(model_cfg, 'num_hidden_layers', 12)
        ffn = getattr(model_cfg, 'intermediate_size', h * 4)
        vocab = getattr(model_cfg, 'vocab_size', 50257)
        # Per layer: attn(4*h*h) + FFN(3*h*ffn for gate/up/down) + norms
        per_layer = 4 * h * h + 3 * h * ffn + 2 * h
        n_params = n_layers * per_layer + vocab * h

    if n_params > 5e9:  # 7B+
        return 2, 8
    elif n_params > 2e9:  # 3B+
        return 4, 16
    else:
        return SPECIALIST_BATCH_SIZE, 32


# ============================================================
# Gram-Schmidt Orthogonalization
# ============================================================

def gram_schmidt(vectors):
    """Orthogonalize a set of vectors using modified Gram-Schmidt.

    Args:
        vectors: np.ndarray of shape (n_vectors, dim)

    Returns:
        orthogonal: np.ndarray of shape (n_vectors, dim) — orthonormal basis
        coefficients: np.ndarray of shape (n_vectors, n_vectors) — projection coefficients
            coefficients[i, j] = how much of original vector j was projected onto basis i
            (useful for understanding shared structure between domains)
    """
    n, d = vectors.shape
    orthogonal = np.zeros_like(vectors, dtype=np.float64)
    coefficients = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        v = vectors[i].astype(np.float64).copy()
        coefficients[i, i] = np.linalg.norm(v)

        for j in range(i):
            proj = np.dot(v, orthogonal[j])
            coefficients[j, i] = proj  # How much of vector i aligns with basis j
            v = v - proj * orthogonal[j]

        norm = np.linalg.norm(v)
        if norm > 1e-10:
            orthogonal[i] = v / norm
        else:
            # Degenerate case: vector lies in span of previous vectors
            orthogonal[i] = v

    return orthogonal.astype(np.float32), coefficients.astype(np.float32)


def measure_basis_orthogonality(centroids, domain_names):
    """Compute pairwise cosine similarity between domain centroids.

    Args:
        centroids: dict mapping domain_name -> np.ndarray (hidden_dim,)
        domain_names: list of domain names in desired order

    Returns:
        dict with pairwise similarities and orthogonality assessment
    """
    names = [d for d in domain_names if d in centroids]
    vecs = np.stack([centroids[d] for d in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    vecs_norm = vecs / norms
    sim_matrix = vecs_norm @ vecs_norm.T

    pairwise = {}
    for i, di in enumerate(names):
        for j, dj in enumerate(names):
            if i < j:
                pairwise[f'{di}-{dj}'] = float(sim_matrix[i, j])

    # Mean off-diagonal similarity
    n = len(names)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_sim = float(sim_matrix[mask].mean()) if mask.any() else 0.0

    return {
        'pairwise_cosine_similarity': pairwise,
        'mean_off_diagonal_similarity': mean_sim,
        'similarity_matrix': sim_matrix.tolist(),
        'domain_order': names,
        'needs_orthogonalization': mean_sim > 0.1,
    }


# ============================================================
# Decomposition-Based Composition
# ============================================================

def decompose_query(query_fp, orthogonal_basis, domain_names):
    """Project query fingerprint onto orthogonalized domain basis.

    The decomposition coefficients directly give composition weights.
    This unifies retrieval and composition into a single operation.

    Args:
        query_fp: np.ndarray of shape (hidden_dim,) — generalist fingerprint on query
        orthogonal_basis: np.ndarray of shape (n_domains, hidden_dim) — orthonormal basis
        domain_names: list of domain names corresponding to basis rows

    Returns:
        dict with:
            'coefficients': dict mapping domain -> float (decomposition coefficient)
            'weights': dict mapping domain -> float (normalized, non-negative)
            'residual_norm': float (energy not captured by basis)
            'sparsity_gini': float (Gini coefficient of |coefficients|)
            'l1_norm': float
            'max_coefficient_domain': str (domain with largest absolute coefficient)
    """
    q = query_fp.astype(np.float64)
    q_norm = q / (np.linalg.norm(q) + 1e-8)

    basis = orthogonal_basis.astype(np.float64)

    # Project query onto each basis vector
    raw_coefficients = {}
    projection = np.zeros_like(q)
    for i, name in enumerate(domain_names):
        coeff = float(np.dot(q_norm, basis[i]))
        raw_coefficients[name] = coeff
        projection += coeff * basis[i]

    # Residual: part of query not explained by any specialist
    residual = q_norm - projection
    residual_norm = float(np.linalg.norm(residual))

    # Sparsity: Gini coefficient of absolute coefficients
    abs_coeffs = np.array([abs(v) for v in raw_coefficients.values()])
    if abs_coeffs.sum() > 1e-10:
        sorted_c = np.sort(abs_coeffs)
        n = len(sorted_c)
        index = np.arange(1, n + 1)
        gini = float((2 * np.sum(index * sorted_c) / (n * sorted_c.sum())) - (n + 1) / n)
    else:
        gini = 0.0

    # Normalized non-negative weights for composition
    # ReLU then normalize — negative coefficients mean anti-correlation
    positive = {k: max(v, 0.0) for k, v in raw_coefficients.items()}
    total = sum(positive.values())
    if total > 1e-8:
        weights = {k: v / total for k, v in positive.items()}
    else:
        # Uniform fallback
        weights = {k: 1.0 / len(domain_names) for k in domain_names}

    max_coeff_domain = max(raw_coefficients, key=lambda k: abs(raw_coefficients[k]))

    return {
        'coefficients': raw_coefficients,
        'weights': weights,
        'residual_norm': residual_norm,
        'sparsity_gini': gini,
        'l1_norm': float(abs_coeffs.sum()),
        'max_coefficient_domain': max_coeff_domain,
    }


def compose_from_decomposition(generalist_sd, specialist_sds, weights, compose_layers=None):
    """Compose specialist parameters using decomposition-derived weights.

    Uses scalar per-specialist weights derived from the decomposition.
    The decomposition already captures which specialists are relevant;
    no second round of activation-based weighting needed.

    θ_composed = Σ w_i * θ_specialist_i  (for composed layers)
    θ_composed = θ_generalist             (for other layers)

    Args:
        generalist_sd: state_dict of generalist model
        specialist_sds: dict mapping specialist_id -> state_dict
        weights: dict mapping specialist_id -> float (sum to 1)
        compose_layers: layer nums to compose (None = last 4)

    Returns:
        composed_sd: new state_dict
    """
    spec_ids = [sid for sid in weights if sid in specialist_sds and weights[sid] > 1e-6]
    composed_sd = {k: v.clone() for k, v in generalist_sd.items()}

    if not spec_ids:
        return composed_sd

    # Determine composable layers
    if compose_layers is None:
        all_layer_nums = set()
        for name in generalist_sd:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p in ('h', 'layers') and i + 1 < len(parts):
                    try:
                        all_layer_nums.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        if all_layer_nums:
            max_layer = max(all_layer_nums)
            compose_layer_nums = set(range(max(0, max_layer - 3), max_layer + 1))
        else:
            compose_layer_nums = set()
    else:
        compose_layer_nums = set(compose_layers)

    for name, gen_param in generalist_sd.items():
        should_compose = False
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p in ('h', 'layers') and i + 1 < len(parts):
                try:
                    if int(parts[i + 1]) in compose_layer_nums:
                        should_compose = True
                except ValueError:
                    pass

        if not should_compose:
            continue
        if not all(name in specialist_sds[sid] for sid in spec_ids):
            continue

        # Weighted sum of specialist parameters (auto-move to gen_param device)
        composed = torch.zeros_like(gen_param, dtype=torch.float32)
        for sid in spec_ids:
            w = weights[sid]
            composed += w * specialist_sds[sid][name].to(composed.device).float()

        composed_sd[name] = composed.to(gen_param.dtype)

    return composed_sd


def partial_fallback_blend(generalist_sd, composed_sd, gen_loss, composed_loss):
    """Blend composed and generalist by loss ratio instead of hard switching.

    If composed_loss is X% worse than gen_loss, use (1-X) composed + X generalist.
    If composed is better, use composed entirely.

    Returns:
        blended_sd: state_dict (may be composed_sd unchanged if composed is better)
        blend_ratio: float (0 = pure composed, 1 = pure generalist)
    """
    if composed_loss <= gen_loss:
        return composed_sd, 0.0

    # How much worse is composed?
    excess = (composed_loss - gen_loss) / max(gen_loss, 1e-8)
    blend_ratio = min(excess, 1.0)

    if blend_ratio < 0.01:
        return composed_sd, 0.0

    blended_sd = {}
    for name in composed_sd:
        if name in generalist_sd:
            blended_sd[name] = (
                (1.0 - blend_ratio) * composed_sd[name].float() +
                blend_ratio * generalist_sd[name].float()
            ).to(composed_sd[name].dtype)
        else:
            blended_sd[name] = composed_sd[name]

    return blended_sd, blend_ratio


# ============================================================
# Specialist Training
# ============================================================

def train_specialist(domain, generalist_ckpt_path, model_cfg, tokenizer,
                     output_dir, device='cpu', use_amp=False,
                     max_steps=SPECIALIST_TRAIN_STEPS):
    """Fine-tune a specialist from the generalist checkpoint on domain data."""
    print(f'\nTraining {domain} specialist...')

    ckpt = torch.load(generalist_ckpt_path, weights_only=False, map_location='cpu')
    gen_sd = {k: v.float() for k, v in ckpt['state_dict'].items()}

    model = _get_model_cls()(model_cfg)
    model.load_state_dict(gen_sd)
    model.to(device)
    del gen_sd

    input_ids, attention_mask = load_domain_data(domain, tokenizer)
    dataset = SimpleTokenDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=_TRAIN_BS,
                            shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SPECIALIST_LR, weight_decay=0.01)

    if device not in ('cuda',) and not str(device).startswith('cuda:'):
        use_amp = False
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    step = 0
    data_iter = iter(dataloader)
    t0 = time.time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        lr = get_lr(step, max_steps, SPECIALIST_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        device_type = device.split(':')[0] if isinstance(device, str) else device.type
        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'])
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        if step % 200 == 0:
            print(f'  [{domain}] Step {step}/{max_steps} | Loss: {loss.item():.4f} | '
                  f'LR: {lr:.2e} | {time.time()-t0:.0f}s')

    spec_dir = output_dir / domain
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_sd = {k: v.cpu().half() for k, v in model.state_dict().items()}
    torch.save({
        'state_dict': spec_sd,
        'domain': domain,
        'steps': max_steps,
        'final_loss': loss.item(),
    }, spec_dir / 'specialist.pt')

    elapsed = time.time() - t0
    print(f'  [{domain}] Done in {elapsed:.0f}s. Final loss: {loss.item():.4f}')

    del model, optimizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return spec_sd


# ============================================================
# Generalist Training
# ============================================================

def train_generalist(model_cfg, tokenizer, output_dir, device='cpu', use_amp=False,
                     max_steps=2000):
    """Train generalist on general data and save checkpoint."""
    print('\nTraining generalist...')

    model = _get_model_cls()(model_cfg)
    model.to(device)

    input_ids, attention_mask = load_domain_data('general', tokenizer)
    dataset = SimpleTokenDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=_TRAIN_BS,
                            shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=SPECIALIST_LR, weight_decay=0.01)

    if device not in ('cuda',) and not str(device).startswith('cuda:'):
        use_amp = False
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(device, enabled=(use_amp and amp_dtype == torch.float16))

    model.train()
    step = 0
    data_iter = iter(dataloader)
    t0 = time.time()

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        lr = get_lr(step, max_steps, SPECIALIST_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        device_type = device.split(':')[0] if isinstance(device, str) else device.type
        with torch.amp.autocast(device_type=device_type, enabled=use_amp, dtype=amp_dtype):
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'])
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        if step % 200 == 0:
            print(f'  [generalist] Step {step}/{max_steps} | Loss: {loss.item():.4f} | '
                  f'LR: {lr:.2e} | {time.time()-t0:.0f}s')

    ckpt_path = output_dir / f'step_{max_steps:05d}.pt'
    gen_sd = {k: v.cpu().half() for k, v in model.state_dict().items()}
    torch.save({
        'state_dict': gen_sd,
        'step': max_steps,
        'final_loss': loss.item(),
    }, ckpt_path)

    elapsed = time.time() - t0
    print(f'  [generalist] Done in {elapsed:.0f}s. Final loss: {loss.item():.4f}')
    print(f'  Saved: {ckpt_path}')

    del model, optimizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return str(ckpt_path)


# ============================================================
# Phase 1: Generalist-Space Index Construction
# ============================================================

def build_generalist_index(generalist_sd, model_cfg, probe_set, device='cpu'):
    """Build domain index in generalist activation space.

    KEY DIFFERENCE FROM PAPER 3: We run ALL probes through the GENERALIST model.
    Domain centroids are the mean generalist fingerprint for each domain's probes.
    This fixes the alignment problem by construction.

    Returns:
        centroids: dict mapping domain -> np.ndarray (hidden_dim,)
        per_probe_fps: dict mapping domain -> np.ndarray (n_probes, hidden_dim)
        all_fps: np.ndarray (total_probes, hidden_dim)
    """
    print('\n' + '=' * 60)
    print('PHASE 1: Generalist-Space Index Construction')
    print('=' * 60)

    model = _get_model_cls()(model_cfg)
    model.load_state_dict({k: v.float() for k, v in generalist_sd.items()})
    model.to(device)

    t0 = time.time()
    fp = capture_at_checkpoint(model, probe_set, ['final_hidden'],
                                device=str(device), batch_size=_CAPTURE_BS)
    all_fps = fp['final_hidden']  # (total_probes, hidden_dim)
    print(f'  Captured generalist fingerprints: {all_fps.shape} in {time.time()-t0:.1f}s')

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    centroids = {}
    per_probe_fps = {}
    for d_idx, domain in enumerate(DOMAINS):
        if domain == 'general':
            continue
        start = d_idx * PROBES_PER_DOMAIN
        end = start + PROBES_PER_DOMAIN
        domain_fps = all_fps[start:end]
        per_probe_fps[domain] = domain_fps
        centroids[domain] = domain_fps.mean(axis=0)
        print(f'  {domain}: centroid shape {centroids[domain].shape}, '
              f'intra-sim {_mean_cosine_sim(domain_fps):.4f}')

    return centroids, per_probe_fps, all_fps


def _mean_cosine_sim(X):
    """Mean pairwise cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n = X / np.maximum(norms, 1e-8)
    sim = X_n @ X_n.T
    n = len(sim)
    if n < 2:
        return 1.0
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return float(sim[mask].mean())


# ============================================================
# Phase 2: Basis Orthogonalization
# ============================================================

def orthogonalize_basis(centroids, domain_order):
    """Measure orthogonality and apply Gram-Schmidt.

    Returns:
        ortho_basis: np.ndarray (n_domains, hidden_dim) — orthonormal
        ortho_info: dict with diagnostics
    """
    print('\n' + '=' * 60)
    print('PHASE 2: Basis Orthogonalization')
    print('=' * 60)

    ortho_info = measure_basis_orthogonality(centroids, domain_order)
    print(f'  Mean off-diagonal similarity (raw): {ortho_info["mean_off_diagonal_similarity"]:.4f}')
    for pair, sim in ortho_info['pairwise_cosine_similarity'].items():
        print(f'    {pair}: {sim:.4f}')

    vectors = np.stack([centroids[d] for d in domain_order])
    ortho_basis, gs_coefficients = gram_schmidt(vectors)

    # Verify
    post_sim = ortho_basis @ ortho_basis.T
    n = len(domain_order)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    post_mean = float(post_sim[mask].mean()) if mask.any() else 0.0
    print(f'  Mean off-diagonal similarity (orthogonalized): {post_mean:.6f}')

    ortho_info['post_orthogonalization_mean_similarity'] = post_mean
    ortho_info['gram_schmidt_coefficients'] = gs_coefficients.tolist()

    return ortho_basis, ortho_info


# ============================================================
# Phase 3: Decomposition-Based Evaluation
# ============================================================

def run_evaluation(generalist_sd, specialist_sds, model_cfg, probe_set,
                   ortho_basis, domain_order, centroids, tokenizer,
                   device='cpu', run_cross_domain=True):
    """Full evaluation: single-domain + cross-domain probes.

    For each probe:
      1. Get generalist fingerprint
      2. Decompose onto orthogonalized specialist basis
      3. Use decomposition weights for composition
      4. Evaluate composed vs generalist vs task arithmetic
      5. Apply partial fallback
      6. Log decomposition coefficients for architecture invariance
    """
    print('\n' + '=' * 60)
    print('PHASE 3: Decomposition-Based Evaluation')
    print('=' * 60)

    # All state dicts stay on CPU. Only the model lives on GPU.
    # Composition is done by directly modifying model parameters in-place on GPU,
    # streaming specialist weights layer-by-layer. No full-dict copies on GPU.
    gen_sd_float = generalist_sd   # CPU half
    spec_sds_float = specialist_sds  # CPU half
    print(f'  State dicts on CPU, model on {device}. In-place composition on GPU.')

    def _load_sd(model, sd):
        """Load state dict into model param-by-param to avoid peak memory spike.
        Uses copy_ which auto-casts dtype and handles cross-device."""
        with torch.no_grad():
            params = dict(model.named_parameters())
            buffers = dict(model.named_buffers())
            for k, v in sd.items():
                if k in params:
                    params[k].data.copy_(v)
                elif k in buffers:
                    buffers[k].copy_(v)

    # Pre-compute composable layer info once (reused every probe)
    _compose_layer_nums = set()
    _all_layer_nums = set()
    for name in generalist_sd:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p in ('h', 'layers') and i + 1 < len(parts):
                try:
                    _all_layer_nums.add(int(parts[i + 1]))
                except ValueError:
                    pass
    if _all_layer_nums:
        _max_layer = max(_all_layer_nums)
        _compose_layer_nums = set(range(max(0, _max_layer - 3), _max_layer + 1))

    # Pre-compute which params are composable (avoid re-parsing every probe)
    _composable_params = set()
    for name in generalist_sd:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p in ('h', 'layers') and i + 1 < len(parts):
                try:
                    if int(parts[i + 1]) in _compose_layer_nums:
                        _composable_params.add(name)
                except ValueError:
                    pass

    def _compose_inplace(model, generalist_sd, specialist_sds, weights):
        """Apply decomposition-based composition directly to model params on GPU.
        Only touches composable layers (last 4). Assumes model already has
        generalist weights for non-composable layers (loaded once)."""
        spec_ids = [sid for sid in weights if sid in specialist_sds and weights[sid] > 1e-6]
        if not spec_ids:
            # Restore composable layers to generalist
            params = dict(model.named_parameters())
            with torch.no_grad():
                for name in _composable_params:
                    if name in params:
                        params[name].data.copy_(generalist_sd[name])
            return

        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in _composable_params:
                if name not in params:
                    continue
                if not all(name in specialist_sds[sid] for sid in spec_ids):
                    continue
                target = params[name]
                composed = torch.zeros_like(target)
                for sid in spec_ids:
                    w = weights[sid]
                    composed.add_(specialist_sds[sid][name].to(target.device, dtype=target.dtype), alpha=w)
                target.data.copy_(composed)

    def _task_arithmetic_inplace(model, generalist_sd, specialist_sds, alpha):
        """Apply task arithmetic directly to model params on GPU.
        Only touches composable layers. Assumes generalist base already loaded."""
        spec_ids = list(specialist_sds.keys())
        alphas = {sid: float(alpha) for sid in spec_ids} if isinstance(alpha, (int, float)) else alpha

        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in _composable_params:
                if name not in params:
                    continue
                if not all(name in specialist_sds[sid] for sid in spec_ids):
                    continue
                target = params[name]
                # Reset to generalist first
                target.data.copy_(generalist_sd[name])
                for sid in spec_ids:
                    task_vec = specialist_sds[sid][name].to(target.device, dtype=target.dtype) - target.data
                    target.data.add_(task_vec, alpha=alphas[sid])

    def _blend_inplace(model, generalist_sd, blend_ratio):
        """Blend composable-layer params toward generalist by blend_ratio, in-place on GPU."""
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name in _composable_params:
                if name not in params:
                    continue
                target = params[name]
                gen_val = generalist_sd[name].to(target.device, dtype=target.dtype)
                target.data.mul_(1.0 - blend_ratio).add_(gen_val, alpha=blend_ratio)

    reuse_model = _get_model_cls()(model_cfg)
    reuse_model.to(device)
    _load_sd(reuse_model, gen_sd_float)

    # Generalist fingerprints for domain probes
    gen_fp = capture_at_checkpoint(reuse_model, probe_set, ['final_hidden'],
                                    device=str(device), batch_size=_CAPTURE_BS)
    gen_activations = gen_fp['final_hidden']

    # Generalist baseline losses
    print('  Computing generalist baseline losses...')
    generalist_losses = []
    for i in range(NUM_PROBES):
        val_batch = {
            'input_ids': probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
        }
        generalist_losses.append(compute_val_loss(reuse_model, val_batch))

    # ---- Single-domain evaluation ----
    print('\n  Evaluating single-domain probes...')
    single_results = []
    decomposition_log = []
    t0 = time.time()
    domain_labels = probe_set['domain_labels']

    for i in range(NUM_PROBES):
        true_domain = domain_labels[i]
        query_fp = gen_activations[i]

        # Decompose
        decomp = decompose_query(query_fp, ortho_basis, domain_order)

        # Compose — in-place on GPU, streaming specialist layers
        _compose_inplace(reuse_model, gen_sd_float, spec_sds_float, decomp['weights'])
        val_batch = {
            'input_ids': probe_set['input_ids'][i:i+1].to(device),
            'attention_mask': probe_set['attention_mask'][i:i+1].to(device),
        }
        composed_loss = compute_val_loss(reuse_model, val_batch)

        # Partial fallback — blend model params toward generalist in-place
        gen_loss = generalist_losses[i]
        excess = (composed_loss - gen_loss) / max(gen_loss, 1e-8) if composed_loss > gen_loss else 0.0
        blend_ratio = min(excess, 1.0) if excess > 0.01 else 0.0

        if blend_ratio > 0.01:
            _blend_inplace(reuse_model, gen_sd_float, blend_ratio)
            blended_loss = compute_val_loss(reuse_model, val_batch)
        else:
            blended_loss = composed_loss

        # Task arithmetic baseline — in-place on GPU
        ta_losses = {}
        for ta_alpha in [0.3, 0.5, 1.0]:
            _task_arithmetic_inplace(reuse_model, gen_sd_float, spec_sds_float, alpha=ta_alpha)
            ta_losses[ta_alpha] = compute_val_loss(reuse_model, val_batch)
        best_ta_alpha = min(ta_losses, key=ta_losses.get)
        best_ta_loss = ta_losses[best_ta_alpha]

        # Oracle — only needs composable layers replaced with specialist
        oracle_loss = float('inf')
        if true_domain in spec_sds_float and true_domain != 'general':
            spec_sd = spec_sds_float[true_domain]
            params = dict(reuse_model.named_parameters())
            with torch.no_grad():
                for name in _composable_params:
                    if name in params and name in spec_sd:
                        params[name].data.copy_(spec_sd[name])
            oracle_loss = compute_val_loss(reuse_model, val_batch)

        # Restore composable layers to generalist for next probe
        _compose_inplace(reuse_model, gen_sd_float, {}, {})

        single_results.append({
            'probe_idx': i,
            'true_domain': true_domain,
            'decomposition_coefficients': decomp['coefficients'],
            'decomposition_weights': decomp['weights'],
            'residual_norm': decomp['residual_norm'],
            'sparsity_gini': decomp['sparsity_gini'],
            'l1_norm': decomp['l1_norm'],
            'max_coefficient_domain': decomp['max_coefficient_domain'],
            'generalist_loss': gen_loss,
            'composed_loss': composed_loss,
            'blended_loss': blended_loss,
            'blend_ratio': blend_ratio,
            'best_task_arithmetic_loss': best_ta_loss,
            'best_task_arithmetic_alpha': best_ta_alpha,
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

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (NUM_PROBES - i - 1) / rate if rate > 0 else 0
            print(f'    {i+1}/{NUM_PROBES} probes ({elapsed:.0f}s, ETA {eta:.0f}s)')

    # Single-domain summary
    non_general = [r for r in single_results if r['true_domain'] != 'general']
    single_summary = {
        'n_probes': len(non_general),
        'win_rate_composed_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in non_general])),
        'win_rate_blended_vs_generalist': float(np.mean([r['blended_beats_generalist'] for r in non_general])),
        'win_rate_composed_vs_task_arithmetic': float(np.mean([r['composed_beats_task_arithmetic'] for r in non_general])),
        'win_rate_blended_vs_task_arithmetic': float(np.mean([r['blended_beats_task_arithmetic'] for r in non_general])),
        'mean_composed_loss': float(np.mean([r['composed_loss'] for r in non_general])),
        'mean_blended_loss': float(np.mean([r['blended_loss'] for r in non_general])),
        'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in non_general])),
        'mean_sparsity_gini': float(np.mean([r['sparsity_gini'] for r in non_general])),
        'mean_residual_norm': float(np.mean([r['residual_norm'] for r in non_general])),
        'mean_blend_ratio': float(np.mean([r['blend_ratio'] for r in non_general])),
    }

    # Per-domain breakdown
    for domain in DOMAINS:
        if domain == 'general':
            continue
        d_results = [r for r in non_general if r['true_domain'] == domain]
        if d_results:
            single_summary[f'{domain}_win_rate_vs_gen'] = float(np.mean(
                [r['composed_beats_generalist'] for r in d_results]))
            single_summary[f'{domain}_mean_sparsity'] = float(np.mean(
                [r['sparsity_gini'] for r in d_results]))

    print(f'\n  Single-domain results:')
    print(f'    Composed vs generalist: {single_summary["win_rate_composed_vs_generalist"]:.1%}')
    print(f'    Blended vs generalist:  {single_summary["win_rate_blended_vs_generalist"]:.1%}')
    print(f'    Composed vs task arith: {single_summary["win_rate_composed_vs_task_arithmetic"]:.1%}')
    print(f'    Mean sparsity (Gini):   {single_summary["mean_sparsity_gini"]:.4f}')
    print(f'    Mean residual norm:     {single_summary["mean_residual_norm"]:.4f}')
    print(f'    Mean blend ratio:       {single_summary["mean_blend_ratio"]:.4f}')

    # ---- Cross-domain evaluation ----
    cross_results = []
    cross_decomp_log = []
    cross_summary = {}

    if run_cross_domain:
        print('\n  Evaluating cross-domain probes...')
        probe_list = EXTENDED_CROSS_DOMAIN_PROBES if _USE_EXTENDED_PROBES else CROSS_DOMAIN_PROBES
        print(f'    Using {"extended" if _USE_EXTENDED_PROBES else "built-in"} probes: {len(probe_list)} total')

        texts = [p[0] for p in probe_list]

        encoded = tokenizer(
            texts, padding='max_length', truncation=True,
            max_length=128, return_tensors='pt',
        )
        cross_probe_set = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }

        _load_sd(reuse_model, gen_sd_float)
        cross_fp = capture_at_checkpoint(reuse_model, cross_probe_set, ['final_hidden'],
                                          device=str(device), batch_size=_CAPTURE_BS)
        cross_activations = cross_fp['final_hidden']

        cross_gen_losses = []
        for i in range(len(texts)):
            val_batch = {
                'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
                'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
            }
            cross_gen_losses.append(compute_val_loss(reuse_model, val_batch))

        t0 = time.time()
        for i, (text, (dom_a, dom_b)) in enumerate(probe_list):
            query_fp = cross_activations[i]

            decomp = decompose_query(query_fp, ortho_basis, domain_order)
            w_a = decomp['weights'].get(dom_a, 0.0)
            w_b = decomp['weights'].get(dom_b, 0.0)

            _compose_inplace(reuse_model, gen_sd_float, spec_sds_float, decomp['weights'])
            val_batch = {
                'input_ids': cross_probe_set['input_ids'][i:i+1].to(device),
                'attention_mask': cross_probe_set['attention_mask'][i:i+1].to(device),
            }
            composed_loss = compute_val_loss(reuse_model, val_batch)

            gen_loss = cross_gen_losses[i]
            excess = (composed_loss - gen_loss) / max(gen_loss, 1e-8) if composed_loss > gen_loss else 0.0
            blend_ratio = min(excess, 1.0) if excess > 0.01 else 0.0
            if blend_ratio > 0.01:
                _blend_inplace(reuse_model, gen_sd_float, blend_ratio)
                blended_loss = compute_val_loss(reuse_model, val_batch)
            else:
                blended_loss = composed_loss

            ta_losses = {}
            for ta_alpha in [0.3, 0.5, 1.0]:
                _task_arithmetic_inplace(reuse_model, gen_sd_float, spec_sds_float, alpha=ta_alpha)
                ta_losses[ta_alpha] = compute_val_loss(reuse_model, val_batch)
            best_ta_alpha = min(ta_losses, key=ta_losses.get)
            best_ta_loss = ta_losses[best_ta_alpha]

            specialist_losses = {}
            params = dict(reuse_model.named_parameters())
            for sid in [dom_a, dom_b]:
                if sid in spec_sds_float:
                    spec_sd = spec_sds_float[sid]
                    with torch.no_grad():
                        for name in _composable_params:
                            if name in params and name in spec_sd:
                                params[name].data.copy_(spec_sd[name])
                    specialist_losses[sid] = compute_val_loss(reuse_model, val_batch)
            best_single = min(specialist_losses.values()) if specialist_losses else float('inf')

            # Restore composable layers to generalist
            _compose_inplace(reuse_model, gen_sd_float, {}, {})

            cross_results.append({
                'probe_idx': i,
                'text_preview': text[:80],
                'expected_domains': [dom_a, dom_b],
                'decomposition_coefficients': decomp['coefficients'],
                'decomposition_weights': decomp['weights'],
                'weight_on_domain_a': w_a,
                'weight_on_domain_b': w_b,
                'residual_norm': decomp['residual_norm'],
                'sparsity_gini': decomp['sparsity_gini'],
                'generalist_loss': gen_loss,
                'composed_loss': composed_loss,
                'blended_loss': blended_loss,
                'blend_ratio': blend_ratio,
                'best_task_arithmetic_loss': best_ta_loss,
                'best_task_arithmetic_alpha': best_ta_alpha,
                'specialist_losses': {k: float(v) for k, v in specialist_losses.items()},
                'best_single_specialist_loss': best_single if best_single < float('inf') else None,
                'composed_beats_generalist': composed_loss < gen_loss,
                'blended_beats_generalist': blended_loss < gen_loss,
                'composed_beats_best_single': composed_loss < best_single,
                'composed_beats_task_arithmetic': composed_loss < best_ta_loss,
                'blended_beats_task_arithmetic': blended_loss < best_ta_loss,
            })

            cross_decomp_log.append({
                'probe_idx': i,
                'expected_domains': [dom_a, dom_b],
                'coefficients': decomp['coefficients'],
                'weights': decomp['weights'],
                'sparsity_gini': decomp['sparsity_gini'],
            })

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(probe_list) - i - 1) / rate if rate > 0 else 0
                print(f'    {i+1}/{len(probe_list)} cross-domain ({elapsed:.0f}s, ETA {eta:.0f}s)')

        cross_summary = {
            'n_probes': len(cross_results),
            'win_rate_composed_vs_generalist': float(np.mean([r['composed_beats_generalist'] for r in cross_results])) if cross_results else 0,
            'win_rate_blended_vs_generalist': float(np.mean([r['blended_beats_generalist'] for r in cross_results])) if cross_results else 0,
            'win_rate_composed_vs_best_single': float(np.mean([r['composed_beats_best_single'] for r in cross_results])) if cross_results else 0,
            'win_rate_composed_vs_task_arithmetic': float(np.mean([r['composed_beats_task_arithmetic'] for r in cross_results])) if cross_results else 0,
            'win_rate_blended_vs_task_arithmetic': float(np.mean([r['blended_beats_task_arithmetic'] for r in cross_results])) if cross_results else 0,
            'mean_composed_loss': float(np.mean([r['composed_loss'] for r in cross_results])) if cross_results else 0,
            'mean_blended_loss': float(np.mean([r['blended_loss'] for r in cross_results])) if cross_results else 0,
            'mean_generalist_loss': float(np.mean([r['generalist_loss'] for r in cross_results])) if cross_results else 0,
            'mean_sparsity_gini': float(np.mean([r['sparsity_gini'] for r in cross_results])) if cross_results else 0,
        }

        print(f'\n  Cross-domain results:')
        print(f'    Composed vs generalist: {cross_summary["win_rate_composed_vs_generalist"]:.1%}')
        print(f'    Blended vs generalist:  {cross_summary["win_rate_blended_vs_generalist"]:.1%}')
        print(f'    Composed vs best single: {cross_summary["win_rate_composed_vs_best_single"]:.1%}')
        print(f'    Composed vs task arith:  {cross_summary["win_rate_composed_vs_task_arithmetic"]:.1%}')

    del reuse_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return {
        'single_domain': {
            'evaluations': single_results,
            'summary': single_summary,
        },
        'cross_domain': {
            'evaluations': cross_results,
            'summary': cross_summary,
        },
        'decomposition_log': {
            'single_domain': decomposition_log,
            'cross_domain': cross_decomp_log,
        },
    }


# ============================================================
# Old Method Comparison
# ============================================================

def run_old_method_comparison(generalist_sd, specialist_sds, model_cfg, probe_set,
                               tokenizer, device='cpu'):
    """Run paper3 specialist-space indexing for A/B comparison."""
    print('\n' + '=' * 60)
    print('COMPARISON: Old Method (Specialist-Space Indexing)')
    print('=' * 60)

    from paper3_constellation import (
        run_pass1_index, run_pass2_retrieval,
        run_pass4_composition, run_pass5_cross_domain,
    )

    # Convert generalist to float32 for old-method functions
    gen_sd_float = {k: v.float() for k, v in generalist_sd.items()}

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for domain, sd in specialist_sds.items():
            spec_dir = tmpdir / domain
            spec_dir.mkdir()
            torch.save({'state_dict': {k: v.half() for k, v in sd.items()},
                        'domain': domain}, spec_dir / 'specialist.pt')

        index = run_pass1_index(tmpdir, model_cfg, probe_set, device=device)
        pass2 = run_pass2_retrieval(gen_sd_float, model_cfg, probe_set, index, device=device)
        pass4 = run_pass4_composition(gen_sd_float, tmpdir, model_cfg, probe_set,
                                       index, device=device, normalize='two_stage')
        pass5 = run_pass5_cross_domain(gen_sd_float, tmpdir, model_cfg, index,
                                        tokenizer, device=device, normalize='two_stage')

    return {
        'retrieval': pass2['summary'],
        'composition': pass4['summary'],
        'cross_domain': pass5['summary'],
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paper 3 Refined: Generalist-Space Decomposition Composition')
    parser.add_argument('--model', type=str, default='gpt2',
                        help='HuggingFace model: gpt2, Qwen/Qwen2.5-1.5B, meta-llama/Llama-3.2-1B, etc.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generalist-ckpt', type=str, default=None,
                        help='Path to generalist checkpoint. Trains one if not set.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output dir. Default: results/paper3_refined/{model}/seed{seed}/')
    parser.add_argument('--train-specialists', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--run-all', action='store_true',
                        help='Train + evaluate (full pipeline)')
    parser.add_argument('--skip-old-comparison', action='store_true')
    parser.add_argument('--skip-cross-domain', action='store_true')
    parser.add_argument('--generalist-steps', type=int, default=2000)
    parser.add_argument('--specialist-steps', type=int, default=2000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    import random; random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_safe = args.model.replace('/', '_')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f'results/paper3_refined/{model_safe}/seed{args.seed}')
    output_dir.mkdir(parents=True, exist_ok=True)
    specialist_dirs = output_dir / 'specialists'

    print('Paper 3 Refined: Generalist-Space Decomposition Composition')
    print(f'Model: {args.model}')
    print(f'Device: {args.device}, AMP: {args.use_amp}, Seed: {args.seed}')
    print(f'Output: {output_dir}')

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

    device = args.device

    # Adaptive batch sizing based on model scale
    global _TRAIN_BS, _CAPTURE_BS
    _TRAIN_BS, _CAPTURE_BS = get_batch_sizes(model_cfg)
    if _TRAIN_BS != SPECIALIST_BATCH_SIZE:
        print(f'  Adaptive batch size: train={_TRAIN_BS}, capture={_CAPTURE_BS} '
              f'(default: {SPECIALIST_BATCH_SIZE}, 32)')

    print('\nTokenizing domain probes...')
    probe_set = tokenize_domain_probes(tokenizer)
    print(f'  {len(probe_set["texts"])} probes across {len(DOMAINS)} domains')

    # Generalist checkpoint
    if args.generalist_ckpt:
        gen_ckpt_path = args.generalist_ckpt
    else:
        gen_ckpt_path = output_dir / f'step_{args.generalist_steps:05d}.pt'
        if not gen_ckpt_path.exists():
            gen_ckpt_path = train_generalist(
                model_cfg, tokenizer, output_dir, device=device,
                use_amp=args.use_amp, max_steps=args.generalist_steps)

    print(f'\nLoading generalist: {gen_ckpt_path}')
    gen_ckpt = torch.load(gen_ckpt_path, weights_only=False, map_location='cpu')
    # Keep in half precision on CPU to save RAM; convert to float32 only on GPU
    generalist_sd = {k: v.half() for k, v in gen_ckpt['state_dict'].items()}
    del gen_ckpt
    gc.collect()

    # Train specialists
    if args.train_specialists or args.run_all:
        specialist_dirs.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            if domain == 'general':
                continue
            spec_path = specialist_dirs / domain / 'specialist.pt'
            if spec_path.exists() and not args.train_specialists:
                print(f'  {domain} specialist exists, skipping')
                continue
            train_specialist(domain, gen_ckpt_path, model_cfg, tokenizer,
                             specialist_dirs, device=device, use_amp=args.use_amp,
                             max_steps=args.specialist_steps)

    if args.train_specialists and not args.run_all:
        print('\nSpecialist training complete.')
        return

    # Load specialists (free checkpoint data eagerly to save CPU RAM)
    specialist_sds = {}
    for domain in DOMAINS:
        if domain == 'general':
            continue
        spec_path = specialist_dirs / domain / 'specialist.pt'
        if not spec_path.exists():
            print(f'  WARNING: No specialist for {domain}')
            continue
        ckpt = torch.load(spec_path, weights_only=False, map_location='cpu')
        specialist_sds[domain] = {k: v.half() for k, v in ckpt['state_dict'].items()}
        del ckpt
    gc.collect()

    if not specialist_sds:
        print('ERROR: No specialists found. Run with --train-specialists first.')
        return

    # Phase 1: Generalist-space index
    centroids, per_probe_fps, all_fps = build_generalist_index(
        generalist_sd, model_cfg, probe_set, device=device)

    index_data = {d: centroids[d].tolist() for d in centroids}
    with open(output_dir / 'generalist_index.json', 'w') as f:
        json.dump(index_data, f, indent=2)

    # Phase 2: Orthogonalize
    domain_order = [d for d in DOMAINS if d != 'general' and d in centroids]
    ortho_basis, ortho_info = orthogonalize_basis(centroids, domain_order)

    with open(output_dir / 'orthogonalization.json', 'w') as f:
        json.dump(ortho_info, f, indent=2, default=float)

    # Phase 3: Evaluate
    eval_results = run_evaluation(
        generalist_sd, specialist_sds, model_cfg, probe_set,
        ortho_basis, domain_order, centroids, tokenizer,
        device=device, run_cross_domain=not args.skip_cross_domain)

    with open(output_dir / 'single_domain_results.json', 'w') as f:
        json.dump(eval_results['single_domain'], f, indent=2, default=float)

    if eval_results['cross_domain']['evaluations']:
        with open(output_dir / 'cross_domain_results.json', 'w') as f:
            json.dump(eval_results['cross_domain'], f, indent=2, default=float)

    with open(output_dir / 'decomposition_log.json', 'w') as f:
        json.dump(eval_results['decomposition_log'], f, indent=2, default=float)

    # Old method comparison
    if not args.skip_old_comparison:
        old_results = run_old_method_comparison(
            generalist_sd, specialist_sds, model_cfg, probe_set,
            tokenizer, device=device)

        with open(output_dir / 'old_method_comparison.json', 'w') as f:
            json.dump(old_results, f, indent=2, default=float)

        print('\n' + '=' * 60)
        print('METHOD COMPARISON')
        print('=' * 60)
        new_sd = eval_results['single_domain']['summary']
        old_sd = old_results['composition']
        print(f'  Single-domain vs generalist:')
        print(f'    NEW (decomposition): {new_sd["win_rate_composed_vs_generalist"]:.1%}')
        print(f'    OLD (specialist idx): {old_sd.get("win_rate_vs_generalist", "N/A")}')

        if eval_results['cross_domain']['summary']:
            new_cd = eval_results['cross_domain']['summary']
            old_cd = old_results['cross_domain']
            print(f'  Cross-domain vs generalist:')
            print(f'    NEW (decomposition): {new_cd["win_rate_composed_vs_generalist"]:.1%}')
            print(f'    OLD (specialist idx): {old_cd.get("win_rate_vs_generalist", "N/A")}')

    # Combined summary
    combined = {
        'model': args.model,
        'seed': args.seed,
        'domain_order': domain_order,
        'orthogonalization': ortho_info,
        'single_domain': eval_results['single_domain']['summary'],
        'cross_domain': eval_results['cross_domain']['summary'],
    }
    with open(output_dir / 'paper3_refined_results.json', 'w') as f:
        json.dump(combined, f, indent=2, default=float)

    print('\n' + '=' * 60)
    print('DONE')
    print(f'Results saved to {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
