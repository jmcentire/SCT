#!/usr/bin/env python3
"""Run Phase 0 for a single seed. Called by run_multi_seed.py with retry logic."""
import os, sys, time, json, gc
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'

import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(__file__))

from src.training.instrumented_trainer import (
    TrainingConfig, CheckpointLog, set_seed, get_device, get_lr,
    SimpleTokenDataset,
)
from src.fingerprint.capture import build_probe_set, capture_at_checkpoint
from src.regime.detect import detect_regimes
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

seed = int(sys.argv[1])
MAX_STEPS = 2000
CHECKPOINT_INTERVAL = 50
BATCH_SIZE = 4

output_dir = Path(f'results/phase0/seed{seed}')
output_dir.mkdir(parents=True, exist_ok=True)

config = TrainingConfig(
    max_steps=MAX_STEPS, batch_size=BATCH_SIZE,
    checkpoint_interval=CHECKPOINT_INTERVAL,
    output_dir=str(output_dir), seed=seed,
)

set_seed(seed)
device = get_device()

# Load pre-tokenized data
cached = torch.load('/tmp/phase0_tokenized.pt', weights_only=True)
shared_ids = cached['input_ids']
shared_mask = cached['attention_mask']
del cached

# Load model
sd = torch.load('/tmp/gpt2_local/model.pt', weights_only=True, map_location='cpu')
cfg = AutoConfig.from_pretrained('/tmp/gpt2_local')
model = AutoModelForCausalLM.from_config(cfg)
model.load_state_dict(sd)
del sd
model = model.to(device)

tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token

dataset = SimpleTokenDataset(shared_ids, shared_mask)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

probe_set = build_probe_set(tok, num_probes=config.num_probes,
                            max_length=config.probe_max_length, seed=seed)

capture_layers = config.capture_layers
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                              weight_decay=config.weight_decay)

checkpoint_logs = []
activation_snapshots = {layer: [] for layer in capture_layers}
checkpoint_steps = []
checkpoint_losses = []

model.eval()
initial_acts = capture_at_checkpoint(model, probe_set, capture_layers, device=str(device))
model.train()
for layer in capture_layers:
    if layer in initial_acts:
        activation_snapshots[layer].append(initial_acts[layer])
checkpoint_steps.append(0)
checkpoint_losses.append(float('inf'))

start_time = time.time()
global_step = 0
running_loss = 0.0
loss_count = 0
data_iter = iter(dataloader)

while global_step < config.max_steps:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    out = model(input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['input_ids'].to(device))
    loss = out.loss / config.gradient_accumulation_steps
    loss.backward()
    running_loss += out.loss.item()
    loss_count += 1

    if (global_step + 1) % config.gradient_accumulation_steps == 0:
        lr = get_lr(global_step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    global_step += 1

    if global_step % config.checkpoint_interval == 0:
        avg_loss = running_loss / max(loss_count, 1)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        acts = capture_at_checkpoint(model, probe_set, capture_layers, device=str(device))
        model.train()

        act_stats = {}
        for layer in capture_layers:
            if layer in acts:
                activation_snapshots[layer].append(acts[layer])
                act_stats[layer] = {
                    'mean_norm': float(np.linalg.norm(acts[layer], axis=1).mean()),
                    'std_norm': float(np.linalg.norm(acts[layer], axis=1).std()),
                    'mean_val': float(acts[layer].mean()),
                    'std_val': float(acts[layer].std()),
                }

        checkpoint_steps.append(global_step)
        checkpoint_losses.append(avg_loss)
        log = CheckpointLog(step=global_step, loss=avg_loss, learning_rate=current_lr,
                            elapsed_seconds=elapsed, activation_stats=act_stats)
        checkpoint_logs.append(log)

        if global_step % 500 == 0:
            print(f'  Step {global_step:5d} | Loss: {avg_loss:.4f} | '
                  f'LR: {current_lr:.2e} | Time: {elapsed:.0f}s', flush=True)

        running_loss = 0.0
        loss_count = 0

total_time = time.time() - start_time
print(f'  Complete: {global_step} steps in {total_time:.0f}s')

# Save
meta = {
    'config': asdict(config),
    'checkpoint_logs': [asdict(log) for log in checkpoint_logs],
    'steps': checkpoint_steps,
    'losses': [float(l) for l in checkpoint_losses],
    'probe_texts': probe_set['texts'],
    'total_time': total_time,
}
with open(output_dir / 'training_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

for layer in capture_layers:
    snaps = activation_snapshots[layer]
    if snaps:
        arr = np.stack(snaps, axis=0)
        np.save(output_dir / f'activations_{layer.replace(".", "_")}.npy', arr)

# Regime detection
snapshots = activation_snapshots['final_hidden']
regime_result = detect_regimes(
    snapshots=snapshots, losses=np.array(checkpoint_losses),
    steps=np.array(checkpoint_steps),
    percentile_high=70, percentile_low=30, variance_window=5,
)

regime_data = {
    'steps': regime_result.steps.tolist(),
    'similarities': regime_result.similarities.tolist(),
    'variances': regime_result.variances.tolist(),
    'curvatures': [float(c) if not np.isnan(c) else None for c in regime_result.curvatures],
    'regimes': regime_result.regime_labels,
    'thresholds': regime_result.thresholds,
    'stable_fraction': regime_result.stable_fraction,
    'chaotic_fraction': regime_result.chaotic_fraction,
}
with open(output_dir / 'regime_results.json', 'w') as f:
    json.dump(regime_data, f, indent=2)

print(f'  {regime_result.summary()}')
