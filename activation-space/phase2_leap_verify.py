#!/usr/bin/env python3
"""Phase 2: Speculative Training (Leap+Verify) Experiment.

Demonstrates the core ASC insight: during stable training regimes, weight states
can be *predicted* via linear extrapolation and *verified cheaply* with a single
forward pass, skipping gradient computation entirely.

At each checkpoint (every 50 steps):
1. Capture activation fingerprint, classify regime (online)
2. Evaluate ALL K-step predictions (K=5,10,20,50) and log results
3. If regime is confirmed stable (2 consecutive readings): accept best prediction,
   skip K steps, advance LR schedule, zero optimizer momentum

Uses the phase1_collapse_experiment.py pattern to avoid Python 3.14 segfaults.
"""
import os, sys, time, json, gc, copy
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
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
    output_dir: str = "results/phase2"
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
# Phase 2 Configuration
# ============================================================

K_VALUES = [5, 10, 20, 50]
CHECKPOINT_INTERVAL = 50
SEED = 42

# Averaged thresholds from Phase 0 across 5 seeds (42-46)
# seed42: high=0.99985, low=0.99748
# seed43: high=0.99962, low=0.99793
# seed44: high=0.99943, low=0.99536
# seed45: high=0.99932, low=0.99610
# seed46: high=0.99923, low=0.99447
THRESHOLD_HIGH = np.mean([0.9998489, 0.9996176, 0.9994301, 0.9993165, 0.9992327])  # ~0.99949
THRESHOLD_LOW = np.mean([0.9974827, 0.9979310, 0.9953635, 0.9961008, 0.9944700])   # ~0.99627
CONSECUTIVE_STABLE_REQUIRED = 2  # Require 2 consecutive "stable" readings

# Phase 0 baseline for comparison
PHASE0_BASELINE_LOSS = 1.756  # seed 42 final loss

# Validation set: hold out last 200 samples, use fixed batch of 32
VAL_HOLDOUT = 200
VAL_BATCH_SIZE = 32

output_dir = Path('results/phase2')
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Speculative Prediction Functions
# ============================================================

def apply_linear_prediction(model, prev_sd, current_sd, K, interval):
    """Apply linear weight extrapolation: w_pred = w_current + (K/interval) * delta.

    Args:
        model: Model to modify in-place
        prev_sd: Previous checkpoint state_dict
        current_sd: Current checkpoint state_dict
        K: Number of steps to predict ahead
        interval: Steps between prev and current checkpoints

    Returns:
        backup_sd: Copy of current weights for rollback
    """
    backup_sd = {k: v.clone() for k, v in model.state_dict().items()}
    scale = K / interval

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in prev_sd and name in current_sd:
                delta = current_sd[name] - prev_sd[name]
                param.data.copy_(current_sd[name] + scale * delta)

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
    """Zero optimizer momentum buffers after a weight jump.

    With beta1=0.9, momentum rebuilds within ~10 steps.
    """
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
# Main Experiment
# ============================================================

print('=' * 60)
print('Phase 2: Speculative Training (Leap+Verify)')
print(f'Seed: {SEED}')
print(f'K values: {K_VALUES}')
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

# Fixed validation batch (first 32 of holdout, deterministic)
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
# Training Loop with Speculative Engine
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
prev_checkpoint_sd = None           # Previous checkpoint's state_dict
prev_activation = None              # Previous checkpoint's activation snapshot
consecutive_stable_count = 0        # Consecutive stable regime readings
step = 0                            # Current logical step
total_steps_trained = 0             # Actual gradient steps taken
total_steps_skipped = 0             # Steps skipped via prediction

# Logging
checkpoint_log = []                 # Per-checkpoint data
prediction_log = []                 # Every prediction attempt
loss_curve_steps = []               # For plotting
loss_curve_values = []

running_loss = 0.0
loss_count = 0
di = iter(dataloader)
t0 = time.time()

print(f'--- TRAINING (steps 0→{config.max_steps}) ---', flush=True)

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
    # Checkpoint: regime detection + speculative prediction
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

        # Current checkpoint state_dict (for prediction)
        current_sd = {k: v.clone() for k, v in model.state_dict().items()}

        # Validation loss at current point
        current_val_loss = compute_val_loss(model, val_batch)

        # ============================================================
        # Evaluate ALL K predictions (even in chaotic regimes for logging)
        # ============================================================
        k_results = {}
        best_accepted_k = None
        best_accepted_loss = current_val_loss

        if prev_checkpoint_sd is not None:
            for K in K_VALUES:
                # Skip if K would overshoot max_steps
                if step + K > config.max_steps:
                    k_results[K] = {
                        'predicted_loss': None,
                        'current_loss': current_val_loss,
                        'accepted': False,
                        'reason': 'would_overshoot',
                    }
                    continue

                # Apply prediction
                backup = apply_linear_prediction(model, prev_checkpoint_sd, current_sd, K, CHECKPOINT_INTERVAL)
                predicted_loss = compute_val_loss(model, val_batch)
                restore_weights(model, backup)

                accepted = predicted_loss < current_val_loss
                k_results[K] = {
                    'predicted_loss': predicted_loss,
                    'current_loss': current_val_loss,
                    'delta': predicted_loss - current_val_loss,
                    'accepted': accepted,
                    'reason': 'loss_improved' if accepted else 'loss_worse',
                }

                # Track best accepted K (only if stable confirmed)
                if accepted and is_stable_confirmed and predicted_loss < best_accepted_loss:
                    best_accepted_k = K
                    best_accepted_loss = predicted_loss

                prediction_log.append({
                    'step': step,
                    'K': K,
                    'regime': regime,
                    'similarity': similarity,
                    'is_stable_confirmed': is_stable_confirmed,
                    'current_val_loss': current_val_loss,
                    'predicted_val_loss': predicted_loss,
                    'delta': predicted_loss - current_val_loss,
                    'accepted_by_loss': accepted,
                    'actually_applied': False,  # Updated below if applied
                })

        # ============================================================
        # Apply best prediction if stable confirmed
        # ============================================================
        applied_k = None
        if best_accepted_k is not None:
            applied_k = best_accepted_k
            # Apply the prediction permanently
            apply_linear_prediction(model, prev_checkpoint_sd, current_sd, applied_k, CHECKPOINT_INTERVAL)

            # Advance step counter
            old_step = step
            step += applied_k
            total_steps_skipped += applied_k

            # Advance LR schedule (it's a function of step number, so just update step)
            new_lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr

            # Zero optimizer momentum
            zero_optimizer_momentum(optimizer)

            # Update prediction log: mark the applied one
            for entry in reversed(prediction_log):
                if entry['step'] == old_step and entry['K'] == applied_k:
                    entry['actually_applied'] = True
                    break

            print(f'  Step {old_step:5d} → {step:5d} | LEAP K={applied_k} | '
                  f'Loss: {current_val_loss:.4f} → {best_accepted_loss:.4f} | '
                  f'Regime: {regime} | Sim: {similarity:.6f}', flush=True)
        else:
            if step % 500 == 0 or (similarity is not None and regime == 'stable'):
                print(f'  Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | '
                      f'Regime: {regime} | Sim: {similarity if similarity else "N/A"} | '
                      f'{elapsed:.0f}s', flush=True)

        # Log checkpoint
        checkpoint_log.append({
            'step': step,
            'train_loss': avg_loss,
            'val_loss': current_val_loss,
            'similarity': similarity,
            'regime': regime,
            'consecutive_stable': consecutive_stable_count,
            'is_stable_confirmed': is_stable_confirmed,
            'k_results': {str(k): v for k, v in k_results.items()},
            'applied_k': applied_k,
            'elapsed': elapsed,
        })

        # Update state for next checkpoint
        prev_checkpoint_sd = current_sd  # Already cloned
        prev_activation = current_activation

        running_loss = 0.0
        loss_count = 0

total_time = time.time() - t0

# ============================================================
# Results Analysis
# ============================================================

print(f'\n{"=" * 60}')
print('LEAP+VERIFY RESULTS')
print(f'{"=" * 60}', flush=True)

# Success rate by K × regime
success_by_k_regime = {}
for K in K_VALUES:
    success_by_k_regime[K] = {}
    for r in ['chaotic', 'transition', 'stable', 'unknown']:
        entries = [e for e in prediction_log if e['K'] == K and e['regime'] == r]
        if entries:
            n_accepted = sum(1 for e in entries if e['accepted_by_loss'])
            success_by_k_regime[K][r] = {
                'total': len(entries),
                'accepted': n_accepted,
                'rate': n_accepted / len(entries),
            }

print('\nSuccess Rate by K × Regime:')
print(f'{"K":>5} {"Regime":<12} {"Accepted":>8} {"Total":>6} {"Rate":>8}')
print('-' * 45)
for K in K_VALUES:
    for r in ['chaotic', 'transition', 'stable']:
        if r in success_by_k_regime.get(K, {}):
            d = success_by_k_regime[K][r]
            print(f'{K:>5} {r:<12} {d["accepted"]:>8} {d["total"]:>6} {d["rate"]:>8.1%}')

# Applied predictions summary
applied = [e for e in prediction_log if e['actually_applied']]
print(f'\nPredictions Applied: {len(applied)}')
for e in applied:
    print(f'  Step {e["step"]} → +{e["K"]} steps, '
          f'loss {e["current_val_loss"]:.4f} → {e["predicted_val_loss"]:.4f}')

# Effective speedup
effective_steps = total_steps_trained + total_steps_skipped
speedup = total_steps_skipped / effective_steps if effective_steps > 0 else 0
print(f'\nSteps trained (gradient): {total_steps_trained}')
print(f'Steps skipped (predicted): {total_steps_skipped}')
print(f'Effective total: {effective_steps}')
print(f'Effective speedup: {speedup:.1%}')
print(f'Total time: {total_time:.0f}s')

# Final loss comparison
final_val_loss = compute_val_loss(model, val_batch)
final_train_loss = loss_curve_values[-1] if loss_curve_values else float('inf')
print(f'\nFinal validation loss: {final_val_loss:.4f}')
print(f'Final training loss: {final_train_loss:.4f}')
print(f'Phase 0 baseline (seed 42): {PHASE0_BASELINE_LOSS:.4f}')

# ============================================================
# Save Results
# ============================================================

# Regime timeline
regimes_detected = [c['regime'] for c in checkpoint_log]
regime_steps = [c['step'] for c in checkpoint_log]

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
    },
    'summary': {
        'total_steps_trained': total_steps_trained,
        'total_steps_skipped': total_steps_skipped,
        'effective_total': effective_steps,
        'effective_speedup': float(speedup),
        'final_val_loss': final_val_loss,
        'final_train_loss': final_train_loss,
        'total_time_seconds': total_time,
        'predictions_applied': len(applied),
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
            r: {
                'total': d['total'],
                'accepted': d['accepted'],
                'rate': d['rate'],
            }
            for r, d in regimes.items()
        }
        for K, regimes in success_by_k_regime.items()
    },
    'prediction_log': prediction_log,
    'checkpoint_log': checkpoint_log,
    'applied_predictions': [
        {
            'step': e['step'],
            'K': e['K'],
            'regime': e['regime'],
            'current_loss': e['current_val_loss'],
            'predicted_loss': e['predicted_val_loss'],
        }
        for e in applied
    ],
}

results_path = output_dir / 'leap_verify_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved results to {results_path}')

# Generate plots
print('\nGenerating plots...')
try:
    from src.analysis.plot_phase2 import plot_all
    import matplotlib
    matplotlib.use('Agg')
    plot_all(results, save_dir=str(output_dir))
    print('Plots saved.')
except Exception as e:
    print(f'Plot generation failed: {e}')
    print('Run src/analysis/plot_phase2.py manually after fixing.')

print('\nDONE')
