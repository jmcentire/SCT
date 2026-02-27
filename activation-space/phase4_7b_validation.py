#!/usr/bin/env python3
"""Phase 4: 7B Scale Validation — Replicate Phase 2c on Qwen2.5-7B.

Thin adapter over phase2c_depth_sweep.py (via phase3_scale_validation.py pattern).
Overrides for 7B scale:
1. Model: Qwen/Qwen2.5-7B (requires A100 80GB)
2. Batch size: 1 (memory constrained at 7B)
3. Gradient accumulation steps: 4 (compensate for batch_size=1)
4. Checkpoint rolling window: 5 (7B checkpoints are ~14GB each)
5. Max length: 256 (same as phase2c/phase3)

Usage:
  python phase4_7b_validation.py --seed 42 [--max-steps 2000] [--skip-training] [--cleanup]
  python phase4_7b_validation.py --seed 42 --device cuda --use-amp
"""
import os, sys, time, json, gc, argparse, shutil
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__) or '.')

# Import everything from phase2c except model-specific bits
from phase2c_depth_sweep import (
    TrainingConfig, SimpleTokenDataset, set_seed, get_lr, classify_regime,
    K_SWEEP_VALUES, PREDICTORS, SWEEP_REGIMES, CHECKPOINT_INTERVAL,
    MOMENTUM_SCALES, THRESHOLD_HIGH, THRESHOLD_LOW, CONSECUTIVE_STABLE_REQUIRED,
    VAL_HOLDOUT, VAL_BATCH_SIZE, WORKER_HANDOFF_THRESHOLD,
    CASCADE_CONFIGS, LANDSCAPE_FRACTIONS,
    run_pass1_training, run_pass2_ksweep, run_pass3_cascades, combine_results,
)
from src.fingerprint.capture import build_probe_set

# ============================================================
# Phase 4 Constants
# ============================================================

PHASE4_MODEL_NAME = 'Qwen/Qwen2.5-7B'
PHASE4_BATCH_SIZE = 1
PHASE4_GRADIENT_ACCUMULATION_STEPS = 4
PHASE4_MAX_LENGTH = 256
CHECKPOINT_ROLLING_WINDOW = 5  # 7B checkpoints are ~14GB each — keep only last 5


def phase4_model_factory(cfg):
    """Create Qwen2.5-7B model from pretrained weights.

    Uses bfloat16 to fit in 80GB GPU memory:
      - bf16 model params: ~15GB
      - bf16 optimizer states: ~30GB
      - Activations + overhead: ~20GB
      - Total: ~65GB (fits in A100 80GB)

    fp32 would need ~90GB (30GB model + 60GB optimizer) — OOMs on 80GB.
    """
    return AutoModelForCausalLM.from_pretrained(
        PHASE4_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )


def tokenize_dataset(model_name, max_length=256, cache_dir='/tmp'):
    """Tokenize WikiText-103 with the target model's tokenizer. Cache to disk."""
    safe_name = model_name.replace('/', '_')
    cache_path = Path(cache_dir) / f'phase4_tokenized_{safe_name}.pt'

    if cache_path.exists():
        print(f'Loading cached tokenized data from {cache_path}')
        cached = torch.load(cache_path, weights_only=True)
        return cached['input_ids'], cached['attention_mask']

    print(f'Tokenizing WikiText-103 with {model_name} tokenizer...')
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')

    # Filter empty lines and tokenize
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    print(f'  {len(texts)} non-empty passages')

    # Tokenize in chunks
    all_ids = []
    all_masks = []
    chunk_size = 1000
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i+chunk_size]
        encoded = tokenizer(
            chunk, padding='max_length', truncation=True,
            max_length=max_length, return_tensors='pt',
        )
        all_ids.append(encoded['input_ids'])
        all_masks.append(encoded['attention_mask'])

        if (i // chunk_size) % 10 == 0:
            print(f'  Tokenized {min(i+chunk_size, len(texts))}/{len(texts)}...')

    input_ids = torch.cat(all_ids, dim=0)
    attention_mask = torch.cat(all_masks, dim=0)

    # Save cache
    torch.save({'input_ids': input_ids, 'attention_mask': attention_mask}, cache_path)
    print(f'  Cached {input_ids.shape[0]} samples to {cache_path}')

    return input_ids, attention_mask


def cleanup_old_checkpoints(ckpt_dir, keep_last_n=CHECKPOINT_ROLLING_WINDOW):
    """Delete old checkpoints, keeping only the most recent N.

    At 7B scale, each checkpoint is ~14GB. With keep_last_n=5, disk usage
    stays under ~70GB for checkpoints.
    """
    ckpt_files = sorted(ckpt_dir.glob('step_*.pt'))
    if len(ckpt_files) > keep_last_n:
        to_delete = ckpt_files[:-keep_last_n]
        for f in to_delete:
            size_gb = f.stat().st_size / (1024**3)
            f.unlink()
            print(f'  Freed {size_gb:.1f}GB: {f.name}')


def main():
    parser = argparse.ArgumentParser(description='Phase 4: 7B Scale Validation (Qwen2.5-7B)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip Pass 1 (use existing checkpoints)')
    parser.add_argument('--skip-pass2', action='store_true',
                        help='Skip Pass 2 K-sweep (use existing results)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Delete checkpoints after aggregation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use mixed precision (bf16 on A100)')
    parser.add_argument('--cache-dir', type=str, default='/tmp',
                        help='Directory for tokenized data cache')
    args = parser.parse_args()

    seed = args.seed
    safe_model = PHASE4_MODEL_NAME.replace('/', '_')
    output_dir = Path(f'results/phase4/{safe_model}/seed{seed}')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    use_amp = args.use_amp
    config = TrainingConfig(
        model_name=PHASE4_MODEL_NAME,
        max_steps=args.max_steps,
        batch_size=PHASE4_BATCH_SIZE,
        max_length=PHASE4_MAX_LENGTH,
        seed=seed,
        device=device,
        use_amp=use_amp,
        gradient_accumulation_steps=PHASE4_GRADIENT_ACCUMULATION_STEPS,
        skip_inline_leap=True,  # Save GPU memory — Pass 2 does full K-sweep from disk
        checkpoint_rolling_window=CHECKPOINT_ROLLING_WINDOW,  # ~14GB per ckpt, keep only 5
    )

    print(f'Phase 4: 7B Scale Validation')
    print(f'Model: {PHASE4_MODEL_NAME}')
    print(f'Device: {device}, AMP: {use_amp}')
    print(f'Seed: {seed}, Max steps: {config.max_steps}')
    print(f'Batch size: {config.batch_size}, Grad accum: {config.gradient_accumulation_steps}')
    print(f'Checkpoint rolling window: {CHECKPOINT_ROLLING_WINDOW}')
    print(f'K sweep: {K_SWEEP_VALUES}')
    print(f'Cascade configs: {CASCADE_CONFIGS}')
    print(f'Output: {output_dir}', flush=True)

    # Check GPU memory
    if device == 'cuda' and torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f'GPU: {torch.cuda.get_device_name(0)}, {gpu_mem:.0f}GB')
        if gpu_mem < 70:
            print(f'WARNING: 7B model requires ~80GB GPU memory. '
                  f'Detected {gpu_mem:.0f}GB. May OOM.', flush=True)

    # Tokenize data with Qwen tokenizer
    all_ids, all_mask = tokenize_dataset(PHASE4_MODEL_NAME, max_length=PHASE4_MAX_LENGTH,
                                          cache_dir=args.cache_dir)

    train_ids = all_ids[:-VAL_HOLDOUT]
    train_mask = all_mask[:-VAL_HOLDOUT]
    val_ids = all_ids[-VAL_HOLDOUT:]
    val_mask = all_mask[-VAL_HOLDOUT:]

    # Smaller val batch for 7B model to avoid OOM on logits
    val_bs = min(VAL_BATCH_SIZE, 4)
    val_batch = {
        'input_ids': val_ids[:val_bs],
        'attention_mask': val_mask[:val_bs],
    }

    # Load model config
    print('Loading model config...')
    model_cfg = AutoConfig.from_pretrained(PHASE4_MODEL_NAME, trust_remote_code=True)

    # Load pretrained model once and extract init state_dict
    # Use bfloat16 to match training model dtype (saves ~15GB CPU RAM)
    print('Loading pretrained Qwen2.5-7B for init weights...')
    init_model = AutoModelForCausalLM.from_pretrained(
        PHASE4_MODEL_NAME, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    init_sd = {k: v.clone() for k, v in init_model.state_dict().items()}
    param_count = sum(v.numel() for v in init_sd.values())
    del init_model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    print(f'Init weights: {len(init_sd)} tensors, '
          f'{param_count / 1e9:.1f}B params', flush=True)

    # Build probe set with Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PHASE4_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    canonical_probe_set = build_probe_set(tokenizer, num_probes=100, max_length=128, seed=0)

    # Monkey-patch model_factory in phase2c module so it uses our factory
    import phase2c_depth_sweep
    phase2c_depth_sweep.model_factory = phase4_model_factory

    # Pass 1: Training
    if not args.skip_training:
        run_pass1_training(config, seed, output_dir, model_cfg, init_sd,
                          train_ids, train_mask, val_batch, canonical_probe_set)

        # Rolling checkpoint cleanup — critical at 7B scale (~14GB per checkpoint)
        ckpt_dir = output_dir / 'checkpoints'
        cleanup_old_checkpoints(ckpt_dir, keep_last_n=CHECKPOINT_ROLLING_WINDOW)
    else:
        print('Skipping Pass 1 (--skip-training)')

    del init_sd
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # Pass 2: K-sweep
    if not args.skip_pass2:
        run_pass2_ksweep(config, seed, output_dir, model_cfg, val_batch)
    else:
        print('Skipping Pass 2 (--skip-pass2)')

    # Pass 3: Cascades
    run_pass3_cascades(config, seed, output_dir, model_cfg, train_ids, train_mask, val_batch)

    # Combine
    combined = combine_results(seed, output_dir)

    # Cleanup
    if args.cleanup:
        ckpt_dir = output_dir / 'checkpoints'
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f'Cleaned up {ckpt_dir}')

    # Plots
    print('\nGenerating plots...')
    try:
        from src.analysis.plot_phase2c import plot_single_seed
        import matplotlib
        matplotlib.use('Agg')
        plot_single_seed(combined, save_dir=str(output_dir))
        print('Plots saved.')
    except Exception as e:
        print(f'Plot generation failed: {e}')
        import traceback
        traceback.print_exc()

    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
