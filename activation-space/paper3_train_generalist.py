#!/usr/bin/env python3
"""Train a generalist checkpoint from a pretrained model on WikiText-103.

Produces a checkpoint in the format expected by paper3_constellation.py:
  {state_dict: ..., step: ..., val_loss: ..., train_loss: ...}

Usage:
  python paper3_train_generalist.py --model Qwen/Qwen2.5-1.5B --device cuda --steps 2000
  python paper3_train_generalist.py --model Qwen/Qwen2.5-1.5B --device cuda --steps 200 --smoke-test
"""
import os, sys, time, json, gc, argparse
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__) or '.')
from src.speculative.predictors import compute_val_loss


class SimpleTokenDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }


def tokenize_wikitext(model_name, max_length=256, cache_dir='/tmp'):
    """Tokenize WikiText-103 with the model's tokenizer, cache to disk."""
    safe_name = model_name.replace('/', '_')
    cache_path = Path(cache_dir) / f'paper3_generalist_tokenized_{safe_name}.pt'

    if cache_path.exists():
        print(f'Loading cached data from {cache_path}')
        cached = torch.load(cache_path, weights_only=True)
        return cached['input_ids'], cached['attention_mask']

    print(f'Tokenizing WikiText-103 with {model_name}...')
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('wikitext', 'wikitext-103-v1', split='train')
    texts = [t for t in dataset['text'] if len(t.strip()) > 50]
    print(f'  {len(texts)} passages')

    all_ids, all_masks = [], []
    chunk_size = 5000
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        enc = tokenizer(chunk, padding='max_length', truncation=True,
                        max_length=max_length, return_tensors='pt')
        # Filter out all-padding
        valid = enc['attention_mask'].sum(dim=1) > max_length // 4
        all_ids.append(enc['input_ids'][valid])
        all_masks.append(enc['attention_mask'][valid])
        print(f'  Tokenized {min(start + chunk_size, len(texts))}/{len(texts)}')

    input_ids = torch.cat(all_ids, dim=0)
    attention_mask = torch.cat(all_masks, dim=0)
    print(f'  Total sequences: {len(input_ids)}')

    torch.save({'input_ids': input_ids, 'attention_mask': attention_mask}, cache_path)
    return input_ids, attention_mask


def get_lr(step, max_steps, base_lr, warmup_steps=100):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        safe_name = args.model.replace('/', '_')
        args.output_dir = f'results/paper3_scale/{safe_name}/seed{args.seed}'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f'Paper 3 Generalist Training')
    print(f'  Model: {args.model}')
    print(f'  Device: {args.device}')
    print(f'  Steps: {args.steps}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  LR: {args.lr}')
    print(f'  Seed: {args.seed}')
    print(f'  Output: {output_dir}')

    # Tokenize data
    input_ids, attention_mask = tokenize_wikitext(
        args.model, max_length=args.max_length)

    # Split: last 5% for validation
    n = len(input_ids)
    n_val = max(100, n // 20)
    perm = torch.randperm(n)
    val_ids = input_ids[perm[:n_val]]
    val_mask = attention_mask[perm[:n_val]]
    train_ids = input_ids[perm[n_val:]]
    train_mask = attention_mask[perm[n_val:]]

    train_dataset = SimpleTokenDataset(train_ids, train_mask)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    val_dataset = SimpleTokenDataset(val_ids[:100], val_mask[:100])
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Load model
    print(f'\nLoading {args.model}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(args.device)
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # AMP setup
    use_amp = args.use_amp and args.device.startswith('cuda')
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(args.device, enabled=(use_amp and amp_dtype == torch.float16))

    # Training loop
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    losses = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(args.device) for k, v in batch.items()}
        lr = get_lr(step, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        device_type = args.device.split(':')[0]
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

        losses.append(loss.item())

        if step % 100 == 0 or step == 1:
            avg_loss = np.mean(losses[-100:])
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = (args.steps - step) / steps_per_sec
            print(f'  Step {step}/{args.steps} | Loss: {avg_loss:.4f} | '
                  f'LR: {lr:.2e} | {steps_per_sec:.1f} steps/s | ETA: {eta/60:.0f}m')

    # Compute validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['input_ids'])
            val_losses.append(out.loss.item())
    val_loss = np.mean(val_losses)
    print(f'\n  Final val loss: {val_loss:.4f}')
    print(f'  Final train loss: {np.mean(losses[-100:]):.4f}')
    print(f'  Total time: {time.time() - t0:.0f}s')

    # Save checkpoint in paper3_constellation.py format
    ckpt_path = output_dir / f'step_{args.steps:05d}.pt'
    state_dict = {k: v.cpu().half() for k, v in model.state_dict().items()}
    torch.save({
        'state_dict': state_dict,
        'step': args.steps,
        'val_loss': val_loss,
        'train_loss': np.mean(losses[-100:]),
    }, ckpt_path)
    print(f'  Saved checkpoint: {ckpt_path}')
    print(f'  Checkpoint size: {ckpt_path.stat().st_size / 1e9:.1f} GB')

    # Save training metadata
    meta = {
        'model': args.model,
        'seed': args.seed,
        'steps': args.steps,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_length': args.max_length,
        'val_loss': val_loss,
        'train_loss': float(np.mean(losses[-100:])),
        'total_time_s': time.time() - t0,
        'checkpoint_path': str(ckpt_path),
    }
    with open(output_dir / 'training_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    del model, optimizer
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
