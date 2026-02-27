"""
Constellation-indexed model composition.

Given a generalist model and specialist models, composes specialist parameters
weighted by activation-space correlation with the generalist fingerprint on
each query. Implements the composition operator from Paper 3.
"""

import torch
import numpy as np
from typing import Optional


def compute_correlation_scores(generalist_fp, specialist_index):
    """Compute cosine similarity between generalist query fingerprint and specialist index entries.

    Args:
        generalist_fp: np.ndarray of shape (hidden_dim,) — generalist activation on query
        specialist_index: dict mapping specialist_id -> np.ndarray of shape (n_domains, hidden_dim)

    Returns:
        dict mapping specialist_id -> float (cosine similarity, max across domains)
    """
    g = generalist_fp / (np.linalg.norm(generalist_fp) + 1e-8)
    scores = {}
    for spec_id, index_entries in specialist_index.items():
        # Max similarity across domain entries for this specialist
        best_sim = -1.0
        for entry in index_entries:
            e = entry / (np.linalg.norm(entry) + 1e-8)
            sim = float(np.dot(g, e))
            best_sim = max(best_sim, sim)
        scores[spec_id] = best_sim
    return scores


def get_activation_signature(model, input_ids, attention_mask, device='cpu'):
    """Get per-node activation magnitudes from the final hidden layer.

    Args:
        model: HuggingFace model
        input_ids: (1, seq_len) tensor
        attention_mask: (1, seq_len) tensor
        device: device string

    Returns:
        np.ndarray of shape (hidden_dim,) — mean-pooled activation magnitudes
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        # Final hidden state: (batch, seq, hidden)
        final_hidden = outputs.hidden_states[-1]
        # Mean pool over sequence, abs for magnitude
        signature = final_hidden.mean(dim=1).abs().squeeze(0).float().cpu().numpy()

    return signature


def compute_node_weights(correlation_scores, activation_signatures, normalize='two_stage'):
    """Compute per-node composition weights from correlation scores and activation signatures.

    Args:
        correlation_scores: dict mapping specialist_id -> float
        activation_signatures: dict mapping specialist_id -> np.ndarray (hidden_dim,)
        normalize: 'joint' for single normalization, 'two_stage' for per-node then scale

    Returns:
        dict mapping specialist_id -> np.ndarray (hidden_dim,) of normalized weights
    """
    spec_ids = list(correlation_scores.keys())
    hidden_dim = len(next(iter(activation_signatures.values())))

    # Clamp negative correlations to 0 (irrelevant specialists get zero weight)
    # Raw weights: r_ij = max(c_i, 0) * A_i(q)_j
    raw = {}
    for sid in spec_ids:
        c = max(correlation_scores[sid], 0.0)
        raw[sid] = c * activation_signatures[sid]

    if normalize == 'joint':
        # Normalize across all models and all nodes jointly
        total = sum(raw[sid].sum() for sid in spec_ids)
        if abs(total) < 1e-8:
            # Uniform fallback when all weights are ~0
            n = len(spec_ids)
            weights = {sid: np.ones(hidden_dim) / n for sid in spec_ids}
        else:
            weights = {sid: raw[sid] / total for sid in spec_ids}

    elif normalize == 'two_stage':
        # Stage 1: Normalize per-node across specialists
        # Stage 2: Scale by correlation
        weights = {}
        for sid in spec_ids:
            weights[sid] = np.zeros(hidden_dim)
        for j in range(hidden_dim):
            node_total = sum(raw[sid][j] for sid in spec_ids)
            if abs(node_total) < 1e-8:
                # Uniform fallback for this node
                for sid in spec_ids:
                    weights[sid][j] = 1.0 / len(spec_ids)
            else:
                for sid in spec_ids:
                    weights[sid][j] = raw[sid][j] / node_total
    else:
        raise ValueError(f'Unknown normalization: {normalize}')

    return weights


def compose_parameters(generalist_sd, specialist_sds, node_weights,
                       compose_layers=None, blend_alpha=1.0):
    """Compose specialist parameters using node-level weights.

    Args:
        generalist_sd: state_dict of generalist model
        specialist_sds: dict mapping specialist_id -> state_dict
        node_weights: dict mapping specialist_id -> np.ndarray (hidden_dim,)
        compose_layers: list of layer name prefixes to compose (None = last 4 transformer layers)
        blend_alpha: interpolation with generalist (1.0 = full composition, 0.0 = pure generalist)

    Returns:
        composed_sd: new state_dict with composed parameters
    """
    spec_ids = list(specialist_sds.keys())
    composed_sd = {k: v.clone() for k, v in generalist_sd.items()}

    # Determine which layers to compose
    if compose_layers is None:
        # Auto-detect: compose last 4 transformer layers
        # GPT-2 pattern: 'transformer.h.{N}.*'
        # Qwen pattern: 'model.layers.{N}.*'
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
        compose_layer_nums = None  # Use prefix matching instead

    for name, gen_param in generalist_sd.items():
        # Check if this parameter should be composed
        should_compose = False
        if compose_layers is not None:
            should_compose = any(name.startswith(prefix) for prefix in compose_layers)
        elif compose_layer_nums:
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p in ('h', 'layers') and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        if layer_num in compose_layer_nums:
                            should_compose = True
                    except ValueError:
                        pass

        if not should_compose:
            continue

        # Check all specialists have this parameter
        if not all(name in specialist_sds[sid] for sid in spec_ids):
            continue

        # Compose: weighted sum of specialist parameters
        # Weight applied per output dimension (node)
        composed = torch.zeros_like(gen_param, dtype=torch.float32)

        for sid in spec_ids:
            spec_param = specialist_sds[sid][name].float()
            w = torch.from_numpy(node_weights[sid]).float()

            # Apply weights along the output dimension
            if spec_param.dim() == 2:
                # Weight matrices: (out, in) — weight per output node
                if spec_param.shape[0] == len(w):
                    composed += spec_param * w.unsqueeze(1)
                elif spec_param.shape[1] == len(w):
                    composed += spec_param * w.unsqueeze(0)
                else:
                    # Fallback: uniform weight by correlation score
                    c = node_weights[sid].mean()
                    composed += spec_param * c
            elif spec_param.dim() == 1:
                # Bias/norm: (hidden_dim,) — direct weight
                if len(spec_param) == len(w):
                    composed += spec_param * w
                else:
                    c = node_weights[sid].mean()
                    composed += spec_param * c
            else:
                # Higher-dim: uniform weight
                c = node_weights[sid].mean()
                composed += spec_param * c

        # Blend with generalist
        if blend_alpha < 1.0:
            composed = blend_alpha * composed + (1.0 - blend_alpha) * gen_param.float()

        composed_sd[name] = composed.to(gen_param.dtype)

    return composed_sd


def task_arithmetic_compose(generalist_sd, specialist_sds, alpha=0.3,
                             compose_layers=None):
    """Compose specialists using task arithmetic (Ilharco et al. 2023).

    θ_composed = θ_generalist + Σ α_i (θ_specialist_i - θ_generalist)

    Args:
        generalist_sd: state_dict of generalist model
        specialist_sds: dict mapping specialist_id -> state_dict
        alpha: scaling coefficient per task vector (uniform or dict)
        compose_layers: layer prefixes to compose (None = last 4 layers)

    Returns:
        composed_sd: new state_dict with task-arithmetic composition
    """
    spec_ids = list(specialist_sds.keys())
    composed_sd = {k: v.clone() for k, v in generalist_sd.items()}

    # Parse alpha: scalar or per-specialist dict
    if isinstance(alpha, (int, float)):
        alphas = {sid: float(alpha) for sid in spec_ids}
    else:
        alphas = alpha

    # Determine which layers to compose (reuse logic from compose_parameters)
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
        compose_layer_nums = None

    for name, gen_param in generalist_sd.items():
        should_compose = False
        if compose_layers is not None:
            should_compose = any(name.startswith(prefix) for prefix in compose_layers)
        elif compose_layer_nums:
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

        # Task arithmetic: θ_gen + Σ α_i * (θ_spec_i - θ_gen)
        task_vector_sum = torch.zeros_like(gen_param, dtype=torch.float32)
        for sid in spec_ids:
            spec_param = specialist_sds[sid][name].to(gen_param.device).float()
            task_vector = spec_param - gen_param.float()
            task_vector_sum += alphas[sid] * task_vector

        composed_sd[name] = (gen_param.float() + task_vector_sum).to(gen_param.dtype)

    return composed_sd


def weight_entropy(node_weights):
    """Compute entropy of composition weights (should be >1 bit for meaningful mixing).

    Args:
        node_weights: dict mapping specialist_id -> np.ndarray (hidden_dim,)

    Returns:
        float: mean entropy across nodes (in bits)
    """
    spec_ids = list(node_weights.keys())
    hidden_dim = len(next(iter(node_weights.values())))

    entropies = []
    for j in range(hidden_dim):
        probs = np.array([max(node_weights[sid][j], 1e-10) for sid in spec_ids])
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        entropies.append(entropy)

    return float(np.mean(entropies))
