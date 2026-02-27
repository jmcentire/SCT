#!/usr/bin/env python3
"""Phase 1, Workstream B: Ensemble Collapse-and-Expand Experiment.

Trains a donor run (seed 42) for 1450 steps, saves weights at the collapse
point, then runs 4 "collapsed" seeds (43-46) from the donor weights for the
remaining 550 steps. Compares final losses to Phase 0 baselines.

Uses the test_minimal.py pattern to avoid Python 3.14 segfaults.
"""
import os, sys, time, json, gc
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

# Inlined classes (avoid importing from instrumented_trainer.py)
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
    output_dir: str = "results/phase1"
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

# Safe imports
sys.path.insert(0, os.path.dirname(__file__) or '.')
from src.fingerprint.capture import build_probe_set, capture_at_checkpoint

# Configuration
COLLAPSE_STEP = 1450
TOTAL_STEPS = 2000
DONOR_SEED = 42
RECIPIENT_SEEDS = [43, 44, 45, 46]
CHECKPOINT_INTERVAL = 50
DONOR_WEIGHTS_PATH = '/tmp/phase1_donor_1450.pt'

PHASE0_BASELINES = {42: 1.756, 43: 1.659, 44: 1.722, 45: 1.697, 46: 1.677}

output_dir = Path('results/phase1')
output_dir.mkdir(parents=True, exist_ok=True)

print('='*60)
print('Phase 1: Collapse-and-Expand Experiment')
print(f'Donor: seed {DONOR_SEED}, Collapse at step {COLLAPSE_STEP}')
print(f'Recipients: seeds {RECIPIENT_SEEDS}')
print('='*60, flush=True)

# Load shared resources
print('Loading pre-tokenized data...')
cached = torch.load('/tmp/phase0_tokenized.pt', weights_only=True)
shared_ids = cached['input_ids']
shared_mask = cached['attention_mask']
del cached; gc.collect()
print(f'Data: {shared_ids.shape[0]} samples')

print('Loading model config and weights...')
model_cfg = AutoConfig.from_pretrained('/tmp/gpt2_local')
init_sd = torch.load('/tmp/gpt2_local/model.pt', weights_only=True, map_location='cpu')
gc.collect()
print(f'Weights: {len(init_sd)} tensors')

tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
gc.collect()

# Build shared canonical probe set (seed=0 for comparability across all runs)
canonical_probe_set = build_probe_set(tok, num_probes=100, max_length=128, seed=0)
print('Ready.\n', flush=True)

def train_segment(model, dataset, config, start_step, end_step, seed, probe_set):
    """Train a model from start_step to end_step. Returns loss curve and activations."""
    set_seed(seed)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True,
                           generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    checkpoint_steps = []
    checkpoint_losses = []
    activation_snapshots = []
    running_loss = 0.0
    loss_count = 0
    step = start_step
    di = iter(dataloader)

    t0 = time.time()

    while step < end_step:
        try:
            batch = next(di)
        except StopIteration:
            di = iter(dataloader)
            batch = next(di)

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

        if step % CHECKPOINT_INTERVAL == 0:
            avg_loss = running_loss / max(loss_count, 1)
            elapsed = time.time() - t0

            model.eval()
            acts = capture_at_checkpoint(model, probe_set, ['final_hidden'], device='cpu')
            model.train()

            checkpoint_steps.append(step)
            checkpoint_losses.append(avg_loss)
            activation_snapshots.append(acts['final_hidden'])

            if step % 500 == 0:
                print(f'    Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.0f}s', flush=True)
            running_loss = 0.0
            loss_count = 0

    total_time = time.time() - t0
    return {
        'steps': checkpoint_steps,
        'losses': checkpoint_losses,
        'activations': activation_snapshots,
        'final_loss': checkpoint_losses[-1] if checkpoint_losses else float('inf'),
        'total_time': total_time,
    }

# ============================================================
# PHASE A: Train donor (seed 42) to collapse point
# ============================================================
print(f'--- DONOR TRAINING (seed {DONOR_SEED}, steps 0→{COLLAPSE_STEP}) ---', flush=True)

config = TrainingConfig(max_steps=TOTAL_STEPS, seed=DONOR_SEED)
set_seed(DONOR_SEED)

model = GPT2LMHeadModel(model_cfg)
model.load_state_dict(init_sd)
gc.collect()

dataset = SimpleTokenDataset(shared_ids, shared_mask)

donor_pre = train_segment(model, dataset, config, 0, COLLAPSE_STEP, DONOR_SEED, canonical_probe_set)
print(f'  Donor at step {COLLAPSE_STEP}: loss={donor_pre["final_loss"]:.4f}, time={donor_pre["total_time"]:.0f}s', flush=True)

# Save donor weights at collapse point
torch.save(model.state_dict(), DONOR_WEIGHTS_PATH)
print(f'  Donor weights saved to {DONOR_WEIGHTS_PATH}', flush=True)

# ============================================================
# PHASE B: Continue donor to completion
# ============================================================
print(f'\n--- DONOR CONTINUATION (seed {DONOR_SEED}, steps {COLLAPSE_STEP}→{TOTAL_STEPS}) ---', flush=True)

donor_post = train_segment(model, dataset, config, COLLAPSE_STEP, TOTAL_STEPS, DONOR_SEED, canonical_probe_set)
donor_final_loss = donor_post['final_loss']
print(f'  Donor final loss: {donor_final_loss:.4f}, time={donor_post["total_time"]:.0f}s', flush=True)

del model; gc.collect()

# ============================================================
# PHASE C: Collapsed runs (seeds 43-46 from donor weights)
# ============================================================
collapsed_results = {}
donor_sd = torch.load(DONOR_WEIGHTS_PATH, weights_only=True, map_location='cpu')

for seed in RECIPIENT_SEEDS:
    print(f'\n--- COLLAPSED RUN (seed {seed}, steps {COLLAPSE_STEP}→{TOTAL_STEPS}) ---', flush=True)

    config = TrainingConfig(max_steps=TOTAL_STEPS, seed=seed)
    set_seed(seed)

    model = GPT2LMHeadModel(model_cfg)
    model.load_state_dict(donor_sd)
    gc.collect()

    result = train_segment(model, dataset, config, COLLAPSE_STEP, TOTAL_STEPS, seed, canonical_probe_set)
    collapsed_results[seed] = result
    print(f'  Seed {seed} final loss: {result["final_loss"]:.4f}, time={result["total_time"]:.0f}s', flush=True)

    del model; gc.collect()

# Clean up donor weights
os.remove(DONOR_WEIGHTS_PATH)
print(f'\nCleaned up {DONOR_WEIGHTS_PATH}')

# ============================================================
# RESULTS
# ============================================================
print(f'\n{"="*60}')
print('COLLAPSE EXPERIMENT RESULTS')
print(f'{"="*60}', flush=True)

all_seeds = [DONOR_SEED] + RECIPIENT_SEEDS
baseline_losses = [PHASE0_BASELINES[s] for s in all_seeds]
collapsed_losses = [donor_final_loss] + [collapsed_results[s]['final_loss'] for s in RECIPIENT_SEEDS]

print(f'\n{"Seed":>6} {"Baseline":>10} {"Collapsed":>10} {"Delta":>10}')
print('-' * 40)
for s, b, c in zip(all_seeds, baseline_losses, collapsed_losses):
    delta = c - b
    print(f'{s:>6} {b:>10.4f} {c:>10.4f} {delta:>+10.4f}')

diffs = [c - b for b, c in zip(baseline_losses, collapsed_losses)]
mean_diff = np.mean(diffs)
std_diff = np.std(diffs)

# Paired t-test
from scipy import stats
t_stat, p_val = stats.ttest_1samp(diffs, 0)
print(f'\nMean delta: {mean_diff:+.4f} ± {std_diff:.4f}')
print(f'Paired t-test: t={t_stat:.3f}, p={p_val:.4f}')
print(f'Fidelity: {"PASS (p > 0.05)" if p_val > 0.05 else "FAIL (p <= 0.05)"}')

# Compute savings
total_baseline = len(all_seeds) * TOTAL_STEPS
total_collapsed = 1 * TOTAL_STEPS + len(RECIPIENT_SEEDS) * (TOTAL_STEPS - COLLAPSE_STEP)
savings = 1 - total_collapsed / total_baseline
print(f'\nCompute savings: {savings:.0%}')
print(f'  Baseline: {len(all_seeds)} × {TOTAL_STEPS} = {total_baseline:,} steps')
print(f'  Collapsed: 1 × {TOTAL_STEPS} + {len(RECIPIENT_SEEDS)} × {TOTAL_STEPS - COLLAPSE_STEP} = {total_collapsed:,} steps')

# Save results
post_collapse_curves = {}
for seed in RECIPIENT_SEEDS:
    r = collapsed_results[seed]
    post_collapse_curves[str(seed)] = {
        'steps': r['steps'],
        'losses': [float(l) for l in r['losses']],
    }
post_collapse_curves[str(DONOR_SEED)] = {
    'steps': donor_post['steps'],
    'losses': [float(l) for l in donor_post['losses']],
}

collapse_output = {
    'collapse_step': COLLAPSE_STEP,
    'total_steps': TOTAL_STEPS,
    'donor_seed': DONOR_SEED,
    'seeds': all_seeds,
    'baseline_losses': baseline_losses,
    'collapsed_losses': collapsed_losses,
    'diffs': diffs,
    'mean_diff': float(mean_diff),
    'std_diff': float(std_diff),
    't_stat': float(t_stat),
    'p_value': float(p_val),
    'compute_savings': float(savings),
    'post_collapse_curves': post_collapse_curves,
    'donor_pre_collapse': {
        'steps': donor_pre['steps'],
        'losses': [float(l) for l in donor_pre['losses']],
        'total_time': donor_pre['total_time'],
    },
}

with open(output_dir / 'collapse_results.json', 'w') as f:
    json.dump(collapse_output, f, indent=2)
print(f'\nSaved to {output_dir / "collapse_results.json"}')

# Save post-collapse activations for divergence analysis
for seed in RECIPIENT_SEEDS:
    acts = collapsed_results[seed]['activations']
    if acts:
        arr = np.stack(acts, axis=0)
        np.save(output_dir / f'collapsed_acts_seed{seed}.npy', arr)
donor_acts = donor_post['activations']
if donor_acts:
    arr = np.stack(donor_acts, axis=0)
    np.save(output_dir / f'collapsed_acts_seed{DONOR_SEED}.npy', arr)

print('DONE')
