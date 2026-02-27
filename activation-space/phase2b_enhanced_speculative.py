#!/usr/bin/env python3
"""Phase 2b: Enhanced Speculative Training Experiment.

Extends Phase 2 (linear extrapolation only) with:
1. Momentum-informed predictor — uses AdamW exp_avg/exp_avg_sq for K-step prediction
2. Quadratic predictor — fits parabola to 3 consecutive checkpoints
3. Landscape probing — directional derivative + activation sensitivity → trust radius
4. Simulated parallel workers — demonstrates ASC architecture with handoff verification
5. Prediction cache — stores successful (start, end) pairs for fast-forwarding

Goal: push stable-regime success rates from 17% toward 50%+ and demonstrate the
full speculative architecture, even if parallelism is simulated sequentially.
"""
import os, sys, time, json, gc, copy, math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from transformers import AutoConfig, GPT2LMHeadModel, AutoTokenizer


# ============================================================
# Inlined classes (avoid importing from instrumented_trainer.py)
# ============================================================

@dataclass
class TrainingConfig:
    model_name: str = "gpt2"
    max_steps: int = 2000
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_length: int = 256
    checkpoint_interval: int = 50
    num_probes: int = 100
    probe_max_length: int = 128
    capture_layers: list = field(default_factory=lambda: ['final_hidden'])
    output_dir: str = "results/phase2b"
    seed: int = 42
    gradient_accumulation_steps: int = 1


class SimpleTokenDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.learning_rate * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))


# ============================================================
# Safe imports from src/
# ============================================================
sys.path.insert(0, os.path.dirname(__file__) or '.')
from src.fingerprint.capture import build_probe_set, capture_at_checkpoint
from src.regime.detect import cosine_similarity_batch


# ============================================================
# Phase 2b Configuration
# ============================================================

K_VALUES = [5, 10, 20, 50]
CHECKPOINT_INTERVAL = 50
SEED = 42

# Averaged thresholds from Phase 0 across 5 seeds (42-46)
THRESHOLD_HIGH = np.mean([0.9998489, 0.9996176, 0.9994301, 0.9993165, 0.9992327])  # ~0.99949
THRESHOLD_LOW = np.mean([0.9974827, 0.9979310, 0.9953635, 0.9961008, 0.9944700])   # ~0.99627
CONSECUTIVE_STABLE_REQUIRED = 2

PHASE0_BASELINE_LOSS = 1.756  # seed 42 final loss

VAL_HOLDOUT = 200
VAL_BATCH_SIZE = 32

# Phase 2b-specific configuration
MOMENTUM_SCALES = [0.8, 1.0, 1.2]       # Scale factors for momentum predictor
PROBE_EPSILON = 1e-3                      # Step size for landscape probing
WORKER_CONFIGS = [                        # (predictor_type, K)
    ('momentum', 5),
    ('linear', 5),
    ('momentum', 10),
]
WORKER_HANDOFF_THRESHOLD = 0.05           # Relative L2 distance for handoff acceptance
CACHE_MAX_ENTRIES = 5                     # Max prediction cache entries
CACHE_HIT_THRESHOLD = 0.03               # Relative L2 for cache hit

output_dir = Path('results/phase2b')
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Predictor Functions
# ============================================================

def apply_linear_prediction(model, prev_sd, current_sd, K, interval):
    """Linear weight extrapolation: w_pred = w_current + (K/interval) * delta."""
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    scale = K / interval

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in prev_sd and name in current_sd:
                delta = current_sd[name] - prev_sd[name]
                param.data.copy_(current_sd[name] + scale * delta)

    return backup_sd


def apply_momentum_prediction(model, optimizer, current_sd, K, lr, scale_factor=1.0):
    """Momentum-informed prediction using AdamW exp_avg/exp_avg_sq.

    w_pred = w_current + scale * K * lr * exp_avg / (sqrt(exp_avg_sq) + eps)

    This IS the Adam update direction, extrapolated K steps. exp_avg gives the
    noise-smoothed gradient direction, exp_avg_sq gives per-parameter adaptive scaling.

    Args:
        model: Model to modify in-place
        optimizer: AdamW optimizer with populated state
        current_sd: Current checkpoint state_dict
        K: Steps to predict ahead
        lr: Current learning rate
        scale_factor: Multiplier to account for extrapolation overshoot (0.8-1.2)

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    eps = 1e-8

    with torch.no_grad():
        for name, param in model.named_parameters():
            state = optimizer.state.get(param, {})
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                # Adam update direction: -lr * exp_avg / (sqrt(exp_avg_sq) + eps)
                # Extrapolate K steps with scale factor
                update = exp_avg / (torch.sqrt(exp_avg_sq) + eps)
                param.data.copy_(current_sd[name] - scale_factor * K * lr * update)
            elif name in current_sd:
                # No optimizer state yet — fall back to current weights (no prediction)
                param.data.copy_(current_sd[name])

    return backup_sd


def apply_quadratic_prediction(model, sd_t0, sd_t1, sd_t2, K, interval):
    """Quadratic prediction using 3 consecutive checkpoints.

    Fits a parabola through the 3 weight snapshots:
      v1 = sd_t1 - sd_t0    (velocity at interval 1)
      v2 = sd_t2 - sd_t1    (velocity at interval 2)
      accel = v2 - v1        (acceleration)
      w_pred = sd_t2 + (K/interval) * v2 + 0.5 * (K/interval)^2 * accel

    Acceleration is clamped to 0.5x velocity magnitude to prevent wild extrapolation.

    Args:
        model: Model to modify in-place
        sd_t0, sd_t1, sd_t2: Three consecutive checkpoint state_dicts
        K: Steps to predict ahead
        interval: Steps between checkpoints

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    t = K / interval

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in sd_t0 and name in sd_t1 and name in sd_t2:
                v1 = sd_t1[name] - sd_t0[name]
                v2 = sd_t2[name] - sd_t1[name]
                accel = v2 - v1

                # Clamp acceleration magnitude to 0.5x velocity magnitude
                v2_norm = torch.norm(v2)
                accel_norm = torch.norm(accel)
                if accel_norm > 0 and v2_norm > 0:
                    max_accel = 0.5 * v2_norm
                    if accel_norm > max_accel:
                        accel = accel * (max_accel / accel_norm)

                pred = sd_t2[name] + t * v2 + 0.5 * t * t * accel
                param.data.copy_(pred)

    return backup_sd


def restore_weights(model, backup_sd):
    """Restore model weights from backup."""
    model.load_state_dict(backup_sd)


def compute_val_loss(model, val_batch):
    """Single forward pass on fixed validation batch. Returns scalar loss."""
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=val_batch['input_ids'],
            attention_mask=val_batch['attention_mask'],
            labels=val_batch['input_ids']
        )
    return out.loss.item()


def zero_optimizer_momentum(optimizer):
    """Zero optimizer momentum buffers after a weight jump."""
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'exp_avg' in state:
                state['exp_avg'].zero_()
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].zero_()


def classify_regime(similarity):
    """Classify regime from activation similarity using fixed Phase 0 thresholds."""
    if similarity >= THRESHOLD_HIGH:
        return 'stable'
    elif similarity <= THRESHOLD_LOW:
        return 'chaotic'
    else:
        return 'transition'


# ============================================================
# Landscape Probing
# ============================================================

def compute_landscape_probe(model, val_batch, direction_sd, current_sd, probe_set,
                            epsilon=PROBE_EPSILON):
    """Compute directional derivative and activation sensitivity along a prediction direction.

    Args:
        model: Current model
        val_batch: Validation batch for loss computation
        direction_sd: State dict representing the prediction direction (predicted - current)
        current_sd: Current state dict
        probe_set: Canonical probe set for activation capture
        epsilon: Step size for finite difference

    Returns:
        dict with 'directional_derivative', 'activation_sensitivity', 'trust_radius'
    """
    # Compute direction unit vector (normalized across all parameters)
    direction_tensors = []
    param_names = []
    for name, param in model.named_parameters():
        if name in direction_sd and name in current_sd:
            d = direction_sd[name] - current_sd[name]
            direction_tensors.append(d.flatten())
            param_names.append(name)

    if not direction_tensors:
        return {'directional_derivative': 0.0, 'activation_sensitivity': 1.0, 'trust_radius': 0}

    direction_flat = torch.cat(direction_tensors)
    direction_norm = torch.norm(direction_flat)
    if direction_norm < 1e-12:
        return {'directional_derivative': 0.0, 'activation_sensitivity': 1.0, 'trust_radius': 0}

    # Normalize direction
    scale = epsilon / direction_norm.item()

    # Current loss
    current_loss = compute_val_loss(model, val_batch)

    # Perturbed loss: w + epsilon * d_hat
    backup = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        idx = 0
        for name, param in model.named_parameters():
            if name in direction_sd and name in current_sd:
                d = direction_sd[name] - current_sd[name]
                param.data.add_(scale * d)

    perturbed_loss = compute_val_loss(model, val_batch)

    # Capture activations at perturbed point for sensitivity
    model.eval()
    acts_perturbed = capture_at_checkpoint(model, probe_set, ['final_hidden'], device='cpu')

    # Restore original weights
    model.load_state_dict(backup)

    # Capture activations at original point
    model.eval()
    acts_original = capture_at_checkpoint(model, probe_set, ['final_hidden'], device='cpu')

    # Directional derivative
    directional_derivative = (perturbed_loss - current_loss) / epsilon

    # Activation sensitivity (1 - cosine similarity = how much activations change)
    if 'final_hidden' in acts_original and 'final_hidden' in acts_perturbed:
        per_probe_sim = cosine_similarity_batch(acts_original['final_hidden'],
                                                 acts_perturbed['final_hidden'])
        activation_cosine_sim = float(per_probe_sim.mean())
        activation_sensitivity = 1.0 - activation_cosine_sim
    else:
        activation_sensitivity = 0.5  # Default if capture fails

    # Trust radius: max_K = max(K_VALUES) * clamp(sensitivity / (1 + |derivative|), 0, 1)
    # BUT: Large derivative = bad, high sensitivity (low cosine) = bad
    # We want trust_radius to be HIGH when derivative is SMALL and sensitivity is LOW
    # So: trust_factor = clamp(1 / (1 + |derivative|) * (1 - sensitivity), 0, 1)
    derivative_factor = 1.0 / (1.0 + abs(directional_derivative))
    stability_factor = max(0.0, 1.0 - activation_sensitivity * 100)  # Scale sensitivity
    trust_factor = max(0.0, min(1.0, derivative_factor * stability_factor))
    trust_radius = int(max(K_VALUES) * trust_factor)

    return {
        'directional_derivative': float(directional_derivative),
        'activation_sensitivity': float(activation_sensitivity),
        'activation_cosine_sim': float(activation_cosine_sim) if 'final_hidden' in acts_original else None,
        'trust_radius': trust_radius,
        'trust_factor': float(trust_factor),
    }


# ============================================================
# Worker Simulation (ASC Architecture)
# ============================================================

def simulate_worker(model_cfg, init_sd, optimizer_state, worker_config, data_iterator,
                    dataloader, val_batch, config, current_step):
    """Simulate a speculative worker running ahead from a predicted state.

    Args:
        model_cfg: Model config for creating temp model
        init_sd: State dict to start from (predicted weights)
        optimizer_state: None (workers start fresh optimizer)
        worker_config: (predictor_type, K) tuple
        data_iterator: Current data iterator position
        dataloader: Full dataloader for re-iteration
        val_batch: Validation batch
        config: Training config
        current_step: Current training step

    Returns:
        dict with worker results (end_sd_fp16, end_loss, steps_trained)
    """
    predictor_type, K = worker_config

    # Create temporary model
    temp_model = GPT2LMHeadModel(model_cfg)
    temp_model.load_state_dict(init_sd)

    temp_optimizer = torch.optim.AdamW(
        temp_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Train K steps
    di = data_iterator  # Share data sequence with main thread
    steps_done = 0
    for _ in range(K):
        sim_step = current_step + steps_done
        if sim_step >= config.max_steps:
            break

        try:
            batch = next(di)
        except StopIteration:
            di = iter(dataloader)
            batch = next(di)

        temp_model.train()
        out = temp_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                         labels=batch['input_ids'])
        out.loss.backward()

        lr = get_lr(sim_step, config)
        for pg in temp_optimizer.param_groups:
            pg['lr'] = lr
        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), 1.0)
        temp_optimizer.step()
        temp_optimizer.zero_grad()
        steps_done += 1

    # Record end state
    end_loss = compute_val_loss(temp_model, val_batch)
    end_sd = {k: v.clone() for k, v in temp_model.state_dict().items()}

    # Convert predicted start to fp16 for cache storage
    start_sd_fp16 = {k: v.half() for k, v in init_sd.items()}

    del temp_model, temp_optimizer
    gc.collect()

    return {
        'predictor_type': predictor_type,
        'K': K,
        'steps_trained': steps_done,
        'end_loss': end_loss,
        'end_sd': end_sd,
        'start_sd_fp16': start_sd_fp16,
        'data_iterator': di,
    }


def check_worker_handoff(worker_result, actual_sd, threshold=WORKER_HANDOFF_THRESHOLD):
    """Check if a worker's predicted start is close enough to actual state for handoff.

    Args:
        worker_result: Result from simulate_worker (must have start_sd_fp16)
        actual_sd: Actual model state_dict at this checkpoint
        threshold: Relative L2 distance threshold

    Returns:
        (accepted, relative_l2_distance)
    """
    predicted_start = worker_result['start_sd_fp16']

    sum_sq_diff = 0.0
    sum_sq_actual = 0.0

    for name in actual_sd:
        if name in predicted_start:
            actual = actual_sd[name].float()
            predicted = predicted_start[name].float()
            sum_sq_diff += torch.sum((actual - predicted) ** 2).item()
            sum_sq_actual += torch.sum(actual ** 2).item()

    if sum_sq_actual < 1e-12:
        return False, float('inf')

    relative_l2 = math.sqrt(sum_sq_diff / sum_sq_actual)
    accepted = relative_l2 < threshold
    return accepted, relative_l2


# ============================================================
# Prediction Cache
# ============================================================

class PredictionCache:
    """Cache of successful (start_state, end_state) pairs for fast-forwarding."""

    def __init__(self, max_entries=CACHE_MAX_ENTRIES, hit_threshold=CACHE_HIT_THRESHOLD):
        self.max_entries = max_entries
        self.hit_threshold = hit_threshold
        self.entries = []  # List of (start_sd_fp16, end_sd, metadata)
        self.event_log = []

    def insert(self, start_sd, end_sd, metadata):
        """Insert a successful prediction pair."""
        start_fp16 = {k: v.half() for k, v in start_sd.items()}
        entry = (start_fp16, {k: v.clone() for k, v in end_sd.items()}, metadata)
        self.entries.append(entry)

        # Evict oldest if over capacity
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

        self.event_log.append({
            'type': 'insert',
            'step': metadata.get('step', -1),
            'predictor': metadata.get('predictor', 'unknown'),
            'K': metadata.get('K', 0),
            'num_entries': len(self.entries),
        })

    def lookup(self, current_sd, step):
        """Look up cache for a state close to current_sd.

        Returns:
            (hit, end_sd, metadata) if found, else (False, None, None)
        """
        for start_fp16, end_sd, metadata in self.entries:
            sum_sq_diff = 0.0
            sum_sq_current = 0.0

            for name in current_sd:
                if name in start_fp16:
                    curr = current_sd[name].float()
                    cached = start_fp16[name].float()
                    sum_sq_diff += torch.sum((curr - cached) ** 2).item()
                    sum_sq_current += torch.sum(curr ** 2).item()

            if sum_sq_current < 1e-12:
                continue

            relative_l2 = math.sqrt(sum_sq_diff / sum_sq_current)

            if relative_l2 < self.hit_threshold:
                self.event_log.append({
                    'type': 'hit',
                    'step': step,
                    'distance': relative_l2,
                    'cached_step': metadata.get('step', -1),
                    'cached_K': metadata.get('K', 0),
                })
                return True, end_sd, metadata

        self.event_log.append({
            'type': 'miss',
            'step': step,
            'num_entries': len(self.entries),
        })
        return False, None, None


# ============================================================
# Main Experiment
# ============================================================

print('=' * 60)
print('Phase 2b: Enhanced Speculative Training')
print(f'Seed: {SEED}')
print(f'K values: {K_VALUES}')
print(f'Momentum scales: {MOMENTUM_SCALES}')
print(f'Worker configs: {WORKER_CONFIGS}')
print(f'Thresholds: high={THRESHOLD_HIGH:.6f}, low={THRESHOLD_LOW:.6f}')
print(f'Consecutive stable required: {CONSECUTIVE_STABLE_REQUIRED}')
print('=' * 60, flush=True)

# Load shared resources
print('Loading pre-tokenized data...')
cached = torch.load('/tmp/phase0_tokenized.pt', weights_only=True)
all_ids = cached['input_ids']
all_mask = cached['attention_mask']
del cached; gc.collect()

total_samples = all_ids.shape[0]
train_ids = all_ids[:-VAL_HOLDOUT]
train_mask = all_mask[:-VAL_HOLDOUT]
val_ids = all_ids[-VAL_HOLDOUT:]
val_mask = all_mask[-VAL_HOLDOUT:]

print(f'Total samples: {total_samples}')
print(f'Train: {train_ids.shape[0]}, Val holdout: {val_ids.shape[0]}')

# Fixed validation batch
val_batch = {
    'input_ids': val_ids[:VAL_BATCH_SIZE],
    'attention_mask': val_mask[:VAL_BATCH_SIZE],
}

print('Loading model config and weights...')
model_cfg = AutoConfig.from_pretrained('/tmp/gpt2_local')
init_sd = torch.load('/tmp/gpt2_local/model.pt', weights_only=True, map_location='cpu')
gc.collect()
print(f'Weights: {len(init_sd)} tensors')

tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
gc.collect()

# Build canonical probe set (seed=0 for comparability)
canonical_probe_set = build_probe_set(tok, num_probes=100, max_length=128, seed=0)
print('Ready.\n', flush=True)


# ============================================================
# Training Loop with Enhanced Speculative Engine
# ============================================================

config = TrainingConfig(max_steps=2000, seed=SEED)
set_seed(SEED)

model = GPT2LMHeadModel(model_cfg)
model.load_state_dict(init_sd)
gc.collect()

dataset = SimpleTokenDataset(train_ids, train_mask)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(SEED))
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# State tracking
prev_checkpoint_sd = None           # t-1 checkpoint state_dict
prev_prev_checkpoint_sd = None      # t-2 checkpoint state_dict (for quadratic)
prev_activation = None
consecutive_stable_count = 0
step = 0
total_steps_trained = 0
total_steps_skipped = 0

# Logging
checkpoint_log = []
prediction_log = []
landscape_probe_log = []
worker_results_log = []
loss_curve_steps = []
loss_curve_values = []

# Phase 2b components
prediction_cache = PredictionCache()
pending_workers = []  # Workers from previous checkpoint awaiting verification

running_loss = 0.0
loss_count = 0
di = iter(dataloader)
t0 = time.time()

print(f'--- TRAINING (steps 0->{config.max_steps}) ---', flush=True)

while step < config.max_steps:
    # Normal training step
    try:
        batch = next(di)
    except StopIteration:
        di = iter(dataloader)
        batch = next(di)

    model.train()
    out = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                labels=batch['input_ids'])
    out.loss.backward()
    running_loss += out.loss.item()
    loss_count += 1

    lr = get_lr(step, config)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    step += 1
    total_steps_trained += 1

    # ============================================================
    # Checkpoint: regime detection + enhanced speculative prediction
    # ============================================================
    if step % CHECKPOINT_INTERVAL == 0:
        avg_loss = running_loss / max(loss_count, 1)
        elapsed = time.time() - t0

        loss_curve_steps.append(step)
        loss_curve_values.append(avg_loss)

        # Capture activations for regime detection
        model.eval()
        acts = capture_at_checkpoint(model, canonical_probe_set, ['final_hidden'], device='cpu')
        current_activation = acts['final_hidden']
        model.train()

        # Compute similarity and classify regime
        similarity = None
        regime = 'unknown'
        if prev_activation is not None:
            per_probe_sim = cosine_similarity_batch(prev_activation, current_activation)
            similarity = float(per_probe_sim.mean())
            regime = classify_regime(similarity)

        # Update consecutive stable count
        if regime == 'stable':
            consecutive_stable_count += 1
        else:
            consecutive_stable_count = 0

        is_stable_confirmed = consecutive_stable_count >= CONSECUTIVE_STABLE_REQUIRED

        # Current checkpoint state_dict
        current_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Validation loss at current point
        current_val_loss = compute_val_loss(model, val_batch)

        # ============================================================
        # [NEW] Verify pending workers from previous checkpoint
        # ============================================================
        for w in pending_workers:
            accepted, rel_l2 = check_worker_handoff(w, current_sd)
            worker_results_log.append({
                'step': step,
                'spawn_step': w.get('spawn_step', step - CHECKPOINT_INTERVAL),
                'predictor_type': w['predictor_type'],
                'K': w['K'],
                'relative_l2': rel_l2,
                'handoff_accepted': accepted,
                'end_loss': w['end_loss'],
                'threshold': WORKER_HANDOFF_THRESHOLD,
            })
            if accepted:
                # Log hypothetical savings
                print(f'  Worker handoff ACCEPTED: {w["predictor_type"]} K={w["K"]} '
                      f'L2={rel_l2:.4f} end_loss={w["end_loss"]:.4f}', flush=True)
        pending_workers = []

        # ============================================================
        # [NEW] Cache lookup — fast-forward if hit
        # ============================================================
        cache_hit, cache_end_sd, cache_meta = prediction_cache.lookup(current_sd, step)
        if cache_hit:
            print(f'  Step {step}: CACHE HIT from step {cache_meta.get("step", "?")} '
                  f'K={cache_meta.get("K", "?")}', flush=True)
            # Note: we log but don't apply cache hits automatically (experimental)

        # ============================================================
        # Evaluate ALL predictors x K values
        # ============================================================
        k_results = {}
        best_accepted_k = None
        best_accepted_loss = current_val_loss
        best_accepted_predictor = None
        best_predicted_sd = None

        if prev_checkpoint_sd is not None:
            for K in K_VALUES:
                if step + K > config.max_steps:
                    k_results[f'linear_K{K}'] = {
                        'predicted_loss': None, 'current_loss': current_val_loss,
                        'accepted': False, 'reason': 'would_overshoot',
                        'predictor': 'linear',
                    }
                    continue

                # --- Linear prediction ---
                backup = apply_linear_prediction(model, prev_checkpoint_sd, current_sd, K, CHECKPOINT_INTERVAL)
                predicted_loss_linear = compute_val_loss(model, val_batch)
                predicted_sd_linear = {k: v.clone() for k, v in model.state_dict().items()} if predicted_loss_linear < current_val_loss else None
                restore_weights(model, backup)

                accepted_linear = predicted_loss_linear < current_val_loss
                k_results[f'linear_K{K}'] = {
                    'predicted_loss': predicted_loss_linear,
                    'current_loss': current_val_loss,
                    'delta': predicted_loss_linear - current_val_loss,
                    'accepted': accepted_linear,
                    'predictor': 'linear',
                }

                prediction_log.append({
                    'step': step, 'K': K, 'predictor': 'linear',
                    'regime': regime, 'similarity': similarity,
                    'is_stable_confirmed': is_stable_confirmed,
                    'current_val_loss': current_val_loss,
                    'predicted_val_loss': predicted_loss_linear,
                    'delta': predicted_loss_linear - current_val_loss,
                    'accepted_by_loss': accepted_linear,
                    'actually_applied': False,
                })

                if accepted_linear and is_stable_confirmed and predicted_loss_linear < best_accepted_loss:
                    best_accepted_k = K
                    best_accepted_loss = predicted_loss_linear
                    best_accepted_predictor = 'linear'
                    best_predicted_sd = predicted_sd_linear

                # --- Momentum prediction (best scale) ---
                best_mom_loss = float('inf')
                best_mom_scale = 1.0
                for sc in MOMENTUM_SCALES:
                    backup = apply_momentum_prediction(model, optimizer, current_sd, K, lr, scale_factor=sc)
                    pl = compute_val_loss(model, val_batch)
                    if pl < best_mom_loss:
                        best_mom_loss = pl
                        best_mom_scale = sc
                        if pl < current_val_loss:
                            mom_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()}
                        else:
                            mom_predicted_sd = None
                    restore_weights(model, backup)

                accepted_mom = best_mom_loss < current_val_loss
                k_results[f'momentum_K{K}'] = {
                    'predicted_loss': best_mom_loss,
                    'current_loss': current_val_loss,
                    'delta': best_mom_loss - current_val_loss,
                    'accepted': accepted_mom,
                    'predictor': 'momentum',
                    'best_scale': best_mom_scale,
                }

                prediction_log.append({
                    'step': step, 'K': K, 'predictor': 'momentum',
                    'regime': regime, 'similarity': similarity,
                    'is_stable_confirmed': is_stable_confirmed,
                    'current_val_loss': current_val_loss,
                    'predicted_val_loss': best_mom_loss,
                    'delta': best_mom_loss - current_val_loss,
                    'accepted_by_loss': accepted_mom,
                    'actually_applied': False,
                    'best_scale': best_mom_scale,
                })

                if accepted_mom and is_stable_confirmed and best_mom_loss < best_accepted_loss:
                    best_accepted_k = K
                    best_accepted_loss = best_mom_loss
                    best_accepted_predictor = 'momentum'
                    best_predicted_sd = mom_predicted_sd

                # --- Quadratic prediction (needs 2 previous checkpoints) ---
                if prev_prev_checkpoint_sd is not None:
                    backup = apply_quadratic_prediction(
                        model, prev_prev_checkpoint_sd, prev_checkpoint_sd, current_sd, K, CHECKPOINT_INTERVAL
                    )
                    predicted_loss_quad = compute_val_loss(model, val_batch)
                    quad_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()} if predicted_loss_quad < current_val_loss else None
                    restore_weights(model, backup)

                    accepted_quad = predicted_loss_quad < current_val_loss
                    k_results[f'quadratic_K{K}'] = {
                        'predicted_loss': predicted_loss_quad,
                        'current_loss': current_val_loss,
                        'delta': predicted_loss_quad - current_val_loss,
                        'accepted': accepted_quad,
                        'predictor': 'quadratic',
                    }

                    prediction_log.append({
                        'step': step, 'K': K, 'predictor': 'quadratic',
                        'regime': regime, 'similarity': similarity,
                        'is_stable_confirmed': is_stable_confirmed,
                        'current_val_loss': current_val_loss,
                        'predicted_val_loss': predicted_loss_quad,
                        'delta': predicted_loss_quad - current_val_loss,
                        'accepted_by_loss': accepted_quad,
                        'actually_applied': False,
                    })

                    if accepted_quad and is_stable_confirmed and predicted_loss_quad < best_accepted_loss:
                        best_accepted_k = K
                        best_accepted_loss = predicted_loss_quad
                        best_accepted_predictor = 'quadratic'
                        best_predicted_sd = quad_predicted_sd

        # ============================================================
        # [NEW] Landscape probing — directional derivative + activation sensitivity
        # ============================================================
        probe_result = None
        if best_predicted_sd is not None:
            probe_result = compute_landscape_probe(
                model, val_batch, best_predicted_sd, current_sd,
                canonical_probe_set, epsilon=PROBE_EPSILON
            )
            landscape_probe_log.append({
                'step': step,
                'regime': regime,
                'predictor': best_accepted_predictor,
                'K': best_accepted_k,
                **probe_result,
            })

            # Check trust radius against accepted K
            if probe_result['trust_radius'] < best_accepted_k:
                print(f'  Step {step}: Trust radius {probe_result["trust_radius"]} < K={best_accepted_k}, '
                      f'reducing', flush=True)
                # Find the best K within trust radius
                original_k = best_accepted_k
                best_accepted_k = None
                best_accepted_loss = current_val_loss
                best_accepted_predictor = None
                best_predicted_sd = None

                for key, res in k_results.items():
                    if res.get('accepted') and res.get('predicted_loss') is not None:
                        k_val = int(key.split('K')[1])
                        if k_val <= probe_result['trust_radius'] and res['predicted_loss'] < best_accepted_loss:
                            best_accepted_k = k_val
                            best_accepted_loss = res['predicted_loss']
                            best_accepted_predictor = res['predictor']
                            # Need to recompute predicted_sd — apply prediction again
                            if best_accepted_predictor == 'linear':
                                backup = apply_linear_prediction(
                                    model, prev_checkpoint_sd, current_sd, k_val, CHECKPOINT_INTERVAL
                                )
                                best_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()}
                                restore_weights(model, backup)
                            elif best_accepted_predictor == 'momentum':
                                backup = apply_momentum_prediction(
                                    model, optimizer, current_sd, k_val, lr,
                                    scale_factor=res.get('best_scale', 1.0)
                                )
                                best_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()}
                                restore_weights(model, backup)
                            elif best_accepted_predictor == 'quadratic' and prev_prev_checkpoint_sd is not None:
                                backup = apply_quadratic_prediction(
                                    model, prev_prev_checkpoint_sd, prev_checkpoint_sd,
                                    current_sd, k_val, CHECKPOINT_INTERVAL
                                )
                                best_predicted_sd = {k: v.clone() for k, v in model.state_dict().items()}
                                restore_weights(model, backup)
        elif prev_checkpoint_sd is not None and current_val_loss is not None:
            # Probe along linear direction even if no prediction accepted (for logging)
            backup = apply_linear_prediction(model, prev_checkpoint_sd, current_sd, 5, CHECKPOINT_INTERVAL)
            linear_direction_sd = {k: v.clone() for k, v in model.state_dict().items()}
            restore_weights(model, backup)
            probe_result = compute_landscape_probe(
                model, val_batch, linear_direction_sd, current_sd,
                canonical_probe_set, epsilon=PROBE_EPSILON
            )
            landscape_probe_log.append({
                'step': step,
                'regime': regime,
                'predictor': 'linear_default',
                'K': 5,
                **probe_result,
            })

        # ============================================================
        # Apply best prediction if stable confirmed
        # ============================================================
        applied_k = None
        applied_predictor = None
        if best_accepted_k is not None and is_stable_confirmed:
            applied_k = best_accepted_k
            applied_predictor = best_accepted_predictor

            # Apply the prediction permanently
            if best_predicted_sd is not None:
                model.load_state_dict(best_predicted_sd)
            else:
                # Fallback: re-apply
                if applied_predictor == 'linear':
                    apply_linear_prediction(model, prev_checkpoint_sd, current_sd, applied_k, CHECKPOINT_INTERVAL)
                elif applied_predictor == 'momentum':
                    apply_momentum_prediction(model, optimizer, current_sd, applied_k, lr)
                elif applied_predictor == 'quadratic' and prev_prev_checkpoint_sd is not None:
                    apply_quadratic_prediction(
                        model, prev_prev_checkpoint_sd, prev_checkpoint_sd,
                        current_sd, applied_k, CHECKPOINT_INTERVAL
                    )

            # Advance step counter
            old_step = step
            step += applied_k
            total_steps_skipped += applied_k

            # Advance LR schedule
            new_lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            # Zero optimizer momentum
            zero_optimizer_momentum(optimizer)

            # Update prediction log: mark the applied one
            for entry in reversed(prediction_log):
                if entry['step'] == old_step and entry['K'] == applied_k and entry['predictor'] == applied_predictor:
                    entry['actually_applied'] = True
                    break

            # Insert into cache
            prediction_cache.insert(
                current_sd,
                {k: v.clone() for k, v in model.state_dict().items()},
                {'step': old_step, 'predictor': applied_predictor, 'K': applied_k}
            )

            print(f'  Step {old_step:5d} -> {step:5d} | LEAP {applied_predictor} K={applied_k} | '
                  f'Loss: {current_val_loss:.4f} -> {best_accepted_loss:.4f} | '
                  f'Regime: {regime} | Sim: {similarity:.6f}', flush=True)
        else:
            if step % 500 == 0 or (similarity is not None and regime == 'stable'):
                probe_info = ''
                if probe_result:
                    probe_info = f' | Trust: {probe_result["trust_radius"]}'
                print(f'  Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | '
                      f'Regime: {regime} | Sim: {similarity if similarity else "N/A"}{probe_info} | '
                      f'{elapsed:.0f}s', flush=True)

        # ============================================================
        # [NEW] Spawn simulated workers (stable regimes only)
        # ============================================================
        if is_stable_confirmed and prev_checkpoint_sd is not None:
            for wc in WORKER_CONFIGS:
                pred_type, wk = wc
                if step + wk > config.max_steps:
                    continue

                # Generate predicted start state for worker
                if pred_type == 'momentum':
                    backup = apply_momentum_prediction(model, optimizer, current_sd, wk, lr)
                    worker_start_sd = {k: v.clone() for k, v in model.state_dict().items()}
                    restore_weights(model, backup)
                elif pred_type == 'linear':
                    backup = apply_linear_prediction(model, prev_checkpoint_sd, current_sd, wk, CHECKPOINT_INTERVAL)
                    worker_start_sd = {k: v.clone() for k, v in model.state_dict().items()}
                    restore_weights(model, backup)
                else:
                    continue

                # Simulate worker
                worker_result = simulate_worker(
                    model_cfg, worker_start_sd, None, wc,
                    di, dataloader, val_batch, config, step
                )
                worker_result['spawn_step'] = step
                di = worker_result.pop('data_iterator')  # Advance shared iterator
                pending_workers.append(worker_result)

        # Log checkpoint
        checkpoint_log.append({
            'step': step,
            'train_loss': avg_loss,
            'val_loss': current_val_loss,
            'similarity': similarity,
            'regime': regime,
            'consecutive_stable': consecutive_stable_count,
            'is_stable_confirmed': is_stable_confirmed,
            'k_results': {k: v for k, v in k_results.items()},
            'applied_k': applied_k,
            'applied_predictor': applied_predictor,
            'probe_result': probe_result,
            'elapsed': elapsed,
        })

        # Update state for next checkpoint
        prev_prev_checkpoint_sd = prev_checkpoint_sd  # Shift history
        prev_checkpoint_sd = current_sd
        prev_activation = current_activation

        running_loss = 0.0
        loss_count = 0

total_time = time.time() - t0


# ============================================================
# Results Analysis
# ============================================================

print(f'\n{"=" * 60}')
print('ENHANCED SPECULATIVE TRAINING RESULTS')
print(f'{"=" * 60}', flush=True)

# Success rate by predictor x K x regime
predictor_comparison = {}
for predictor in ['linear', 'momentum', 'quadratic']:
    predictor_comparison[predictor] = {}
    for K in K_VALUES:
        predictor_comparison[predictor][K] = {}
        for r in ['chaotic', 'transition', 'stable', 'unknown']:
            entries = [e for e in prediction_log
                       if e['predictor'] == predictor and e['K'] == K and e['regime'] == r]
            if entries:
                n_accepted = sum(1 for e in entries if e['accepted_by_loss'])
                predictor_comparison[predictor][K][r] = {
                    'total': len(entries),
                    'accepted': n_accepted,
                    'rate': n_accepted / len(entries),
                }

print('\nPredictor Comparison (Success Rate):')
print(f'{"Predictor":<12} {"K":>5} {"Regime":<12} {"Accepted":>8} {"Total":>6} {"Rate":>8}')
print('-' * 55)
for predictor in ['linear', 'momentum', 'quadratic']:
    for K in K_VALUES:
        for r in ['chaotic', 'transition', 'stable']:
            d = predictor_comparison.get(predictor, {}).get(K, {}).get(r)
            if d:
                print(f'{predictor:<12} {K:>5} {r:<12} {d["accepted"]:>8} {d["total"]:>6} {d["rate"]:>8.1%}')

# Determine best predictor
best_predictor = 'linear'
best_rate = 0
for predictor in ['linear', 'momentum', 'quadratic']:
    stable_entries = [e for e in prediction_log if e['predictor'] == predictor and e['regime'] == 'stable']
    if stable_entries:
        rate = sum(1 for e in stable_entries if e['accepted_by_loss']) / len(stable_entries)
        if rate > best_rate:
            best_rate = rate
            best_predictor = predictor

print(f'\nBest predictor (stable regime): {best_predictor} ({best_rate:.1%})')

# Applied predictions summary
applied = [e for e in prediction_log if e['actually_applied']]
print(f'\nPredictions Applied: {len(applied)}')
for e in applied:
    print(f'  Step {e["step"]} -> +{e["K"]} steps ({e["predictor"]}), '
          f'loss {e["current_val_loss"]:.4f} -> {e["predicted_val_loss"]:.4f}')

# Worker simulation results
n_workers_total = len(worker_results_log)
n_workers_accepted = sum(1 for w in worker_results_log if w['handoff_accepted'])
hypothetical_steps_saved = sum(w['K'] for w in worker_results_log if w['handoff_accepted'])

print(f'\nWorker Simulation:')
print(f'  Workers evaluated: {n_workers_total}')
print(f'  Handoffs accepted: {n_workers_accepted}')
print(f'  Hypothetical steps saved: {hypothetical_steps_saved}')

# Effective speedup
effective_steps = total_steps_trained + total_steps_skipped
speedup = total_steps_skipped / effective_steps if effective_steps > 0 else 0
hypothetical_speedup = hypothetical_steps_saved / effective_steps if effective_steps > 0 else 0

print(f'\nSteps trained (gradient): {total_steps_trained}')
print(f'Steps skipped (predicted): {total_steps_skipped}')
print(f'Effective total: {effective_steps}')
print(f'Effective speedup: {speedup:.1%}')
print(f'Hypothetical parallel speedup (workers): {hypothetical_speedup:.1%}')
print(f'Total time: {total_time:.0f}s')

# Cache stats
cache_hits = sum(1 for e in prediction_cache.event_log if e['type'] == 'hit')
cache_inserts = sum(1 for e in prediction_cache.event_log if e['type'] == 'insert')
print(f'\nPrediction Cache: {cache_inserts} inserts, {cache_hits} hits')

# Landscape probing stats
if landscape_probe_log:
    derivatives = [p['directional_derivative'] for p in landscape_probe_log]
    trust_radii = [p['trust_radius'] for p in landscape_probe_log]
    print(f'\nLandscape Probing ({len(landscape_probe_log)} probes):')
    print(f'  Directional derivative: mean={np.mean(derivatives):.4f}, '
          f'std={np.std(derivatives):.4f}')
    print(f'  Trust radius: mean={np.mean(trust_radii):.1f}, '
          f'min={min(trust_radii)}, max={max(trust_radii)}')

# Final loss
final_val_loss = compute_val_loss(model, val_batch)
final_train_loss = loss_curve_values[-1] if loss_curve_values else float('inf')
print(f'\nFinal validation loss: {final_val_loss:.4f}')
print(f'Final training loss: {final_train_loss:.4f}')
print(f'Phase 0 baseline (seed 42): {PHASE0_BASELINE_LOSS:.4f}')


# ============================================================
# Save Results
# ============================================================

regimes_detected = [c['regime'] for c in checkpoint_log]
regime_steps = [c['step'] for c in checkpoint_log]

# Build success_by_k_regime for backward compatibility with Phase 2 plotting
success_by_k_regime = {}
for K in K_VALUES:
    success_by_k_regime[K] = {}
    for r in ['chaotic', 'transition', 'stable', 'unknown']:
        # Aggregate across all predictors for backward compat
        entries = [e for e in prediction_log if e['K'] == K and e['regime'] == r]
        if entries:
            n_accepted = sum(1 for e in entries if e['accepted_by_loss'])
            success_by_k_regime[K][r] = {
                'total': len(entries),
                'accepted': n_accepted,
                'rate': n_accepted / len(entries),
            }

results = {
    'config': {
        'seed': SEED,
        'max_steps': config.max_steps,
        'checkpoint_interval': CHECKPOINT_INTERVAL,
        'k_values': K_VALUES,
        'threshold_high': float(THRESHOLD_HIGH),
        'threshold_low': float(THRESHOLD_LOW),
        'consecutive_stable_required': CONSECUTIVE_STABLE_REQUIRED,
        'val_holdout': VAL_HOLDOUT,
        'val_batch_size': VAL_BATCH_SIZE,
        'phase0_baseline_loss': PHASE0_BASELINE_LOSS,
        'momentum_scales': MOMENTUM_SCALES,
        'worker_configs': WORKER_CONFIGS,
        'worker_handoff_threshold': WORKER_HANDOFF_THRESHOLD,
        'cache_max_entries': CACHE_MAX_ENTRIES,
        'cache_hit_threshold': CACHE_HIT_THRESHOLD,
        'probe_epsilon': PROBE_EPSILON,
    },
    'summary': {
        'total_steps_trained': total_steps_trained,
        'total_steps_skipped': total_steps_skipped,
        'effective_total': effective_steps,
        'effective_speedup': float(speedup),
        'hypothetical_parallel_speedup': float(hypothetical_speedup),
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'total_time_seconds': total_time,
        'predictions_applied': len(applied),
        'best_predictor': best_predictor,
        'best_predictor_stable_rate': float(best_rate),
        'workers_evaluated': n_workers_total,
        'workers_accepted': n_workers_accepted,
        'hypothetical_steps_saved': hypothetical_steps_saved,
        'cache_hits': cache_hits,
        'cache_inserts': cache_inserts,
    },
    'loss_curve': {
        'steps': loss_curve_steps,
        'values': [float(v) for v in loss_curve_values],
    },
    'regime_timeline': {
        'steps': regime_steps,
        'regimes': regimes_detected,
        'similarities': [c['similarity'] for c in checkpoint_log],
    },
    'success_by_k_regime': {
        str(K): {
            r: {'total': d['total'], 'accepted': d['accepted'], 'rate': d['rate']}
            for r, d in regimes.items()
        }
        for K, regimes in success_by_k_regime.items()
    },
    'predictor_comparison': {
        pred: {
            str(K): {
                r: {'total': d['total'], 'accepted': d['accepted'], 'rate': d['rate']}
                for r, d in regimes.items()
            }
            for K, regimes in k_data.items()
        }
        for pred, k_data in predictor_comparison.items()
    },
    'prediction_log': prediction_log,
    'checkpoint_log': [
        {k: v for k, v in c.items() if k != 'probe_result'}
        | ({'probe_result': c['probe_result']} if c.get('probe_result') else {})
        for c in checkpoint_log
    ],
    'applied_predictions': [
        {
            'step': e['step'], 'K': e['K'], 'predictor': e['predictor'],
            'regime': e['regime'],
            'current_loss': e['current_val_loss'],
            'predicted_loss': e['predicted_val_loss'],
        }
        for e in applied
    ],
    'landscape_probe_log': landscape_probe_log,
    'worker_results_log': worker_results_log,
    'cache_event_log': prediction_cache.event_log,
}

results_path = output_dir / 'enhanced_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved results to {results_path}')

# Generate plots
print('\nGenerating plots...')
try:
    from src.analysis.plot_phase2b import plot_all
    import matplotlib
    matplotlib.use('Agg')
    plot_all(results, save_dir=str(output_dir))
    print('Plots saved.')
except Exception as e:
    print(f'Plot generation failed: {e}')
    import traceback
    traceback.print_exc()
    print('Run src/analysis/plot_phase2b.py manually after fixing.')

print('\nDONE')
