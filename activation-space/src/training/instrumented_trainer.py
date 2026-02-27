"""
Instrumented training loop for Phase 0: Regime Detection.

Wraps a standard causal LM fine-tuning loop with checkpoint hooks that:
1. Run a fixed probe set through the model
2. Capture activation vectors at specified layers
3. Log loss, learning rate, and activation statistics
4. Save everything for post-hoc regime analysis

Designed to run on Colab free tier (T4 GPU) with GPT-2 small.
"""

import os
# Prevent segfaults from HuggingFace tokenizers parallelism on Python 3.14 + macOS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from pathlib import Path

# These imports are from our own modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.fingerprint.capture import ActivationCapture, build_probe_set, capture_at_checkpoint


@dataclass
class TrainingConfig:
    """Configuration for instrumented training."""
    model_name: str = "gpt2"
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-103-v1"
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
    output_dir: str = "results/phase0"
    seed: int = 42
    gradient_accumulation_steps: int = 1


@dataclass
class CheckpointLog:
    """Logged data at each checkpoint."""
    step: int
    loss: float
    learning_rate: float
    elapsed_seconds: float
    activation_stats: dict  # per-layer mean norm, std


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS on Apple Silicon: skip for training due to segfault issues
    # with GPT-2 Conv1D backward pass on MPS in PyTorch 2.x.
    # CPU on M1 Max is fast enough for GPT-2 small.
    return torch.device('cpu')


def load_model_and_tokenizer(model_name: str):
    """Load a causal LM and its tokenizer.

    Uses a local PyTorch checkpoint cache to avoid intermittent segfaults
    from safetensors deserialization on Python 3.14 + macOS.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading from local PyTorch cache first (avoids safetensors crash)
    cache_dir = f'/tmp/{model_name.replace("/", "_")}_local'
    model_path = os.path.join(cache_dir, 'model.pt')
    if os.path.exists(model_path):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cache_dir)
        model = AutoModelForCausalLM.from_config(config)
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Cache locally for next time
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        model.config.save_pretrained(cache_dir)

    return model, tokenizer


class SimpleTokenDataset(torch.utils.data.Dataset):
    """Simple in-memory tokenized dataset to avoid HF datasets memory-mapping issues."""
    def __init__(self, input_ids_list, attention_mask_list):
        self.input_ids = input_ids_list
        self.attention_mask = attention_mask_list

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
        }


def _download_wikitext_raw(cache_dir: str = '/tmp/wikitext_cache') -> list[str]:
    """Download wikitext-103 raw text without using HF datasets library.

    Uses huggingface_hub to download parquet files and pyarrow to read them,
    completely bypassing the HF datasets library (which crashes on Python 3.14).
    Falls back to generating synthetic training data if download fails.
    """
    os.makedirs(cache_dir, exist_ok=True)
    txt_path = os.path.join(cache_dir, 'wiki.train.texts.txt')

    if os.path.exists(txt_path):
        print(f"  Loading cached WikiText-103 from {txt_path}...")
        texts = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        print(f"  Loaded {len(texts)} texts from cache")
        return texts

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        print("  Downloading WikiText-103 from HuggingFace Hub...")
        parquet_files = [
            'wikitext-103-v1/train-00000-of-00002.parquet',
            'wikitext-103-v1/train-00001-of-00002.parquet',
        ]
        texts = []
        for pf in parquet_files:
            local_path = hf_hub_download(
                repo_id='Salesforce/wikitext',
                filename=pf,
                repo_type='dataset',
                cache_dir=cache_dir,
            )
            table = pq.read_table(local_path)
            for row in table.to_pydict()['text']:
                line = row.strip()
                if line and not line.startswith('='):
                    texts.append(line)
            print(f"    {pf}: {len(texts)} texts so far")

        # Cache to plain text for fast reload
        with open(txt_path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write(t + '\n')
        print(f"  WikiText-103 ready: {len(texts)} texts (cached to {txt_path})")
        return texts
    except Exception as e:
        print(f"  WikiText download failed ({e}), generating synthetic data...")
        return _generate_synthetic_texts(10000)


def _generate_synthetic_texts(n: int) -> list[str]:
    """Generate diverse synthetic texts for training if download fails."""
    rng = np.random.RandomState(42)
    templates = [
        "The history of {topic} dates back to ancient times when people first began to {action}.",
        "In modern computing, {topic} plays a crucial role in how we {action} data and information.",
        "Scientists have discovered that {topic} can be explained through the principles of {field}.",
        "The economic impact of {topic} has been studied extensively by researchers in {field}.",
        "Understanding {topic} requires a deep knowledge of {field} and its applications.",
    ]
    topics = ["mathematics", "physics", "biology", "chemistry", "computer science",
              "literature", "philosophy", "economics", "music", "architecture"]
    actions = ["understand", "analyze", "process", "transform", "evaluate",
               "interpret", "synthesize", "categorize", "optimize", "develop"]
    fields = ["quantum mechanics", "information theory", "statistical analysis",
              "machine learning", "cognitive science", "number theory"]

    texts = []
    for i in range(n):
        t = rng.choice(templates)
        t = t.replace("{topic}", rng.choice(topics))
        t = t.replace("{action}", rng.choice(actions))
        t = t.replace("{field}", rng.choice(fields))
        texts.append(t)
    return texts


def load_dataset_tokens(tokenizer, config: TrainingConfig):
    """Load and tokenize training data. Bypasses HF datasets to avoid
    Python 3.14 + Arrow segfaults on macOS."""

    print("  Loading text data (bypassing HF datasets)...")
    texts = _download_wikitext_raw()

    # Only keep enough data for training
    needed = int(config.max_steps * config.batch_size * 1.5)
    if len(texts) > needed:
        texts = texts[:needed]

    print(f"  Tokenizing {len(texts)} texts...")

    # Tokenize in batches and store as plain tensors
    all_input_ids = []
    all_attention_mask = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=config.max_length,
            padding='max_length',
            return_tensors='pt',
        )
        all_input_ids.append(enc['input_ids'])
        all_attention_mask.append(enc['attention_mask'])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    print(f"  Dataset ready: {input_ids.shape[0]} samples, max_length={input_ids.shape[1]}")

    return SimpleTokenDataset(input_ids, attention_mask)


def get_lr(step: int, config: TrainingConfig) -> float:
    """Linear warmup then cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    return config.learning_rate * max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))


def run_training(config: TrainingConfig, verbose: bool = True):
    """
    Run instrumented training and return all captured data.

    Returns:
        dict with keys:
            'config': TrainingConfig as dict
            'checkpoint_logs': list of CheckpointLog dicts
            'activation_snapshots': dict mapping layer_name -> list of numpy arrays
            'steps': list of checkpoint step numbers
            'losses': list of loss values at checkpoints
            'probe_texts': list of probe input strings
    """
    set_seed(config.seed)
    device = get_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Device: {device}")
        print(f"Loading model: {config.model_name}")

    model, tokenizer = load_model_and_tokenizer(config.model_name)
    model = model.to(device)
    model.train()

    if verbose:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        print(f"Loading dataset: {config.dataset_name}/{config.dataset_config}")

    dataset = load_dataset_tokens(tokenizer, config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if verbose:
        print(f"Building probe set ({config.num_probes} probes)")

    probe_set = build_probe_set(
        tokenizer,
        num_probes=config.num_probes,
        max_length=config.probe_max_length,
        seed=config.seed,
    )

    # Resolve layer names for this model architecture
    capture_layers = config.capture_layers

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Storage
    checkpoint_logs = []
    activation_snapshots = {layer: [] for layer in capture_layers}
    checkpoint_steps = []
    checkpoint_losses = []

    # Capture initial state (step 0)
    if verbose:
        print("Capturing initial activations (step 0)...")
    initial_acts = capture_at_checkpoint(
        model, probe_set, capture_layers, device=str(device)
    )
    for layer in capture_layers:
        if layer in initial_acts:
            activation_snapshots[layer].append(initial_acts[layer])
    checkpoint_steps.append(0)
    checkpoint_losses.append(float('inf'))  # No loss at step 0

    # Training loop
    start_time = time.time()
    global_step = 0
    running_loss = 0.0
    loss_count = 0
    data_iter = iter(dataloader)

    if verbose:
        print(f"\nStarting training for {config.max_steps} steps "
              f"(checkpoint every {config.checkpoint_interval})...\n")

    while global_step < config.max_steps:
        # Get next batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # Forward pass — causal LM uses input as both input and target
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss / config.gradient_accumulation_steps
        loss.backward()

        running_loss += outputs.loss.item()
        loss_count += 1

        if (global_step + 1) % config.gradient_accumulation_steps == 0:
            # Update learning rate
            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        # Checkpoint
        if global_step % config.checkpoint_interval == 0:
            avg_loss = running_loss / max(loss_count, 1)
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']

            # Capture activations
            model.eval()
            acts = capture_at_checkpoint(
                model, probe_set, capture_layers, device=str(device)
            )
            model.train()

            # Compute activation statistics
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

            log = CheckpointLog(
                step=global_step,
                loss=avg_loss,
                learning_rate=current_lr,
                elapsed_seconds=elapsed,
                activation_stats=act_stats,
            )
            checkpoint_logs.append(log)

            if verbose:
                print(f"Step {global_step:5d} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {elapsed:.0f}s")

            running_loss = 0.0
            loss_count = 0

    total_time = time.time() - start_time
    if verbose:
        print(f"\nTraining complete. {global_step} steps in {total_time:.0f}s")
        print(f"Captured {len(checkpoint_steps)} checkpoints")

    # Package results
    results = {
        'config': asdict(config),
        'checkpoint_logs': [asdict(log) for log in checkpoint_logs],
        'steps': np.array(checkpoint_steps),
        'losses': np.array(checkpoint_losses),
        'probe_texts': probe_set['texts'],
        'activation_snapshots': activation_snapshots,
        'total_time': total_time,
    }

    # Save metadata (not the big activation arrays)
    meta = {
        'config': asdict(config),
        'checkpoint_logs': [asdict(log) for log in checkpoint_logs],
        'steps': checkpoint_steps,
        'losses': [float(l) for l in checkpoint_losses],
        'probe_texts': probe_set['texts'],
        'total_time': total_time,
    }
    meta_path = output_dir / 'training_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"Metadata saved to {meta_path}")

    # Save activation snapshots as numpy
    for layer in capture_layers:
        snaps = activation_snapshots[layer]
        if snaps:
            arr = np.stack(snaps, axis=0)  # (num_checkpoints, num_probes, hidden_dim)
            snap_path = output_dir / f'activations_{layer.replace(".", "_")}.npy'
            np.save(snap_path, arr)
            if verbose:
                print(f"Activations ({layer}): {arr.shape} saved to {snap_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Phase 0: Instrumented Training')
    parser.add_argument('--model', default='gpt2', help='HuggingFace model name')
    parser.add_argument('--dataset', default='wikitext', help='Dataset name')
    parser.add_argument('--dataset-config', default='wikitext-103-v1', help='Dataset config')
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--checkpoint-interval', type=int, default=50)
    parser.add_argument('--num-probes', type=int, default=100)
    parser.add_argument('--output-dir', default='results/phase0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_interval=args.checkpoint_interval,
        num_probes=args.num_probes,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    run_training(config)
