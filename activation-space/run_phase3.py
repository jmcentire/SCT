#!/usr/bin/env python3
"""Phase 3: Multi-seed wrapper for scale validation.

Usage:
  python run_phase3.py --model Qwen/Qwen2.5-1.5B           # All 5 seeds
  python run_phase3.py --model Qwen/Qwen2.5-1.5B --seeds 42 43
  python run_phase3.py --model Qwen/Qwen2.5-1.5B --aggregate-only
"""

import os, sys, json, subprocess, time, argparse
import numpy as np
from pathlib import Path

# Reuse aggregation logic from run_phase2c
sys.path.insert(0, os.path.dirname(__file__) or '.')
from run_phase2c import aggregate_results as aggregate_phase2c

SEEDS = [42, 43, 44, 45, 46]


def run_seed(seed, model, max_steps=2000, batch_size=4, extra_args=None):
    """Run a single seed as subprocess."""
    cmd = [sys.executable, 'phase3_scale_validation.py',
           '--seed', str(seed), '--model', model,
           '--max-steps', str(max_steps), '--batch-size', str(batch_size)]
    if extra_args:
        cmd.extend(extra_args)

    print(f'\n{"=" * 60}')
    print(f'Running seed {seed} with {model}...')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"=" * 60}', flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.')
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'WARNING: Seed {seed} exited with code {result.returncode}', flush=True)
        return False, elapsed
    else:
        print(f'Seed {seed} completed in {elapsed:.0f}s', flush=True)
        return True, elapsed


def aggregate_phase3(seeds, model, base_dir=None):
    """Aggregate per-seed results. Uses same schema as Phase 2c."""
    safe_model = model.replace('/', '_')
    if base_dir is None:
        base_dir = Path(f'results/phase3/{safe_model}')

    # Reuse phase2c aggregation — same JSON schema
    return aggregate_phase2c(seeds, base_dir=base_dir)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Multi-seed scale validation')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--aggregate-only', action='store_true')
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_phase3(args.seeds, args.model)
        return

    extra_args = []
    if args.cleanup:
        extra_args.append('--cleanup')

    results = {}
    for seed in args.seeds:
        success, elapsed = run_seed(seed, args.model, max_steps=args.max_steps,
                                     batch_size=args.batch_size, extra_args=extra_args)
        results[seed] = {'success': success, 'time': elapsed}

    print(f'\n{"=" * 60}')
    print(f'PHASE 3 SEED SUMMARY ({args.model})')
    print(f'{"=" * 60}')
    total_cost_hrs = sum(r['time'] for r in results.values()) / 3600
    for seed, r in results.items():
        status = 'OK' if r['success'] else 'FAIL'
        print(f'  Seed {seed}: {status} ({r["time"]:.0f}s)')
    print(f'  Total GPU hours: {total_cost_hrs:.1f}h')
    print(f'  Estimated cost: ${total_cost_hrs * 1.0:.2f} (at $1/hr)')

    aggregate_phase3(args.seeds, args.model)
    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
