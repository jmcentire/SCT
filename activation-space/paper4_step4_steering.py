"""Paper 4 Step 4a: Steering Transfer.

For each held-out domain, compute structural steering vectors from the other
three domains (domain-erased, shape-averaged), inject them during inference,
and test whether the correct-shape vector produces activations classified as
the correct shape more often than wrong-shape vectors.

This demonstrates that the structural subspace is CAUSALLY ACTIVE, not just
correlational — steering with structure from domain A produces structurally
correct behavior in domain B.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python3 paper4_step4_steering.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent))
from paper4_probes import SHAPES, DOMAINS, SHAPE_PROBES, get_all_probes

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# ============================================================
# Load saved decomposition
# ============================================================

def load_decomposition():
    """Load domain directions and activations from Step 3."""
    out_dir = Path('results/paper4')
    removed_dirs = np.load(out_dir / 'removed_domain_directions.npy')
    activations = np.load(out_dir / 'shape_activations.npy')
    with open(out_dir / 'shape_labels.json') as f:
        data = json.load(f)
    probes = data['probes']
    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])
    return removed_dirs, activations, shape_labels, domain_labels


def domain_erase(activations, domain_directions):
    """Project out domain directions from activation vectors."""
    X = activations.copy()
    for d in domain_directions:
        d = d / (np.linalg.norm(d) + 1e-10)
        X = X - np.outer(X @ d, d)
    return X


def compute_steering_vectors(activations_erased, shape_labels, domain_labels,
                             held_out_domain):
    """Compute per-shape structural steering vectors from non-held-out domains.

    Returns dict: shape -> steering vector (hidden_dim,)
    """
    source_mask = domain_labels != held_out_domain
    vectors = {}
    for shape in SHAPES:
        mask = source_mask & (shape_labels == shape)
        vectors[shape] = activations_erased[mask].mean(axis=0)
    return vectors


# ============================================================
# Steering hooks
# ============================================================

class SteeringHook:
    """Adds a steering vector to hidden states at a specific layer."""

    def __init__(self, steering_vector, layer_idx, alpha=1.0):
        self.steering_vector = steering_vector  # (hidden_dim,) numpy
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.handle = None

    def hook_fn(self, module, input, output):
        # output may be a tuple or a single tensor depending on the layer
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        device = hidden.device
        dtype = hidden.dtype
        sv = torch.tensor(self.steering_vector, device=device, dtype=dtype)
        # Add steering vector to all token positions
        hidden = hidden + self.alpha * sv.unsqueeze(0).unsqueeze(0)

        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        else:
            return hidden

    def register(self, model):
        layer = model.model.layers[self.layer_idx]
        self.handle = layer.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self.handle:
            self.handle.remove()


# ============================================================
# Capture steered activations
# ============================================================

def capture_single(model, tokenizer, text, max_length=128):
    """Forward pass one text, return final hidden state mean-pooled."""
    encoded = tokenizer(
        text, padding='max_length', truncation=True,
        max_length=max_length, return_tensors='pt',
    )
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(
            input_ids=encoded['input_ids'].to(device),
            attention_mask=encoded['attention_mask'].to(device),
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1].float()
        mask = encoded['attention_mask'].to(device).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.cpu().numpy().squeeze()


# ============================================================
# Main experiment
# ============================================================

def main():
    print('Paper 4 Step 4a: Steering Transfer')
    print('=' * 60)

    # Load decomposition
    domain_dirs, raw_activations, shape_labels, domain_labels = load_decomposition()
    print(f'Loaded {len(raw_activations)} activations, {len(domain_dirs)} domain directions')

    # Domain-erase activations
    erased_activations = domain_erase(raw_activations, domain_dirs)

    # Train shape classifier on domain-erased activations (this is our evaluation tool)
    print('\nTraining shape classifier on domain-erased activations...')
    shape_clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    shape_clf.fit(erased_activations, shape_labels)
    train_acc = shape_clf.score(erased_activations, shape_labels)
    print(f'  Shape classifier train accuracy: {train_acc:.3f}')

    # Also train on raw activations for comparison
    shape_clf_raw = LogisticRegression(max_iter=2000, solver='lbfgs')
    shape_clf_raw.fit(raw_activations, shape_labels)

    # Load model
    print('\nLoading Qwen2.5-3B...')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B')
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-3B', dtype=torch.float16,
        device_map='cpu', low_cpu_mem_usage=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    steering_layer = n_layers // 2  # Middle layer
    print(f'  Layers: {n_layers}, steering at layer {steering_layer}')

    # Alpha sweep: find good steering strength
    # Use a small pilot to calibrate
    print('\n--- Alpha calibration ---')
    pilot_text = SHAPE_PROBES['constraint']['legal'][0]
    pilot_sv = compute_steering_vectors(
        erased_activations, shape_labels, domain_labels, 'legal')['constraint']

    baseline_act = capture_single(model, tokenizer, pilot_text)
    baseline_norm = np.linalg.norm(baseline_act)
    sv_norm = np.linalg.norm(pilot_sv)
    print(f'  Activation norm: {baseline_norm:.1f}, steering vector norm: {sv_norm:.1f}')
    print(f'  Ratio: {sv_norm/baseline_norm:.3f}')

    # Test alphas
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
        hook = SteeringHook(pilot_sv, steering_layer, alpha=alpha).register(model)
        steered_act = capture_single(model, tokenizer, pilot_text)
        hook.remove()
        cos_sim = np.dot(baseline_act, steered_act) / (
            np.linalg.norm(baseline_act) * np.linalg.norm(steered_act) + 1e-10)
        shift = np.linalg.norm(steered_act - baseline_act)
        print(f'  alpha={alpha:.1f}: cos_sim={cos_sim:.4f}, shift={shift:.1f}')

    # Main experiment: leave-one-domain-out steering transfer
    print('\n' + '=' * 60)
    print('STEERING TRANSFER EXPERIMENT')
    print('=' * 60)

    # Use moderate alpha
    alpha = 2.0
    print(f'\nAlpha: {alpha}')

    results_by_target = {}
    all_correct = 0
    all_wrong = 0
    all_none = 0
    all_total = 0

    for target_domain in DOMAINS:
        print(f'\n--- Target domain: {target_domain} ---')

        # Compute steering vectors from other three domains
        steering_vecs = compute_steering_vectors(
            erased_activations, shape_labels, domain_labels, target_domain)

        target_probes = []
        for shape in SHAPES:
            for text in SHAPE_PROBES[shape][target_domain]:
                target_probes.append((shape, text))

        # For each target probe, test:
        # 1. No steering (baseline)
        # 2. Correct-shape steering
        # 3. Each wrong-shape steering
        correct_shape_wins = 0
        total = 0
        detail_rows = []

        for true_shape, text in target_probes:
            # Baseline (no steering)
            act_baseline = capture_single(model, tokenizer, text)
            act_baseline_erased = domain_erase(act_baseline.reshape(1, -1), domain_dirs).squeeze()
            pred_baseline = shape_clf.predict(act_baseline_erased.reshape(1, -1))[0]
            proba_baseline = shape_clf.predict_proba(act_baseline_erased.reshape(1, -1))[0]

            # Correct-shape steering
            hook = SteeringHook(steering_vecs[true_shape], steering_layer, alpha=alpha).register(model)
            act_correct = capture_single(model, tokenizer, text)
            hook.remove()
            act_correct_erased = domain_erase(act_correct.reshape(1, -1), domain_dirs).squeeze()
            pred_correct = shape_clf.predict(act_correct_erased.reshape(1, -1))[0]
            proba_correct = shape_clf.predict_proba(act_correct_erased.reshape(1, -1))[0]

            # Wrong-shape steering (average across all wrong shapes)
            wrong_preds = []
            wrong_probas = []
            for wrong_shape in SHAPES:
                if wrong_shape == true_shape:
                    continue
                hook = SteeringHook(steering_vecs[wrong_shape], steering_layer, alpha=alpha).register(model)
                act_wrong = capture_single(model, tokenizer, text)
                hook.remove()
                act_wrong_erased = domain_erase(act_wrong.reshape(1, -1), domain_dirs).squeeze()
                pred_wrong = shape_clf.predict(act_wrong_erased.reshape(1, -1))[0]
                proba_wrong = shape_clf.predict_proba(act_wrong_erased.reshape(1, -1))[0]
                wrong_preds.append(pred_wrong)
                wrong_probas.append(proba_wrong)

            # Score: probability assigned to the TRUE shape class
            shape_idx = list(shape_clf.classes_).index(true_shape)
            p_baseline = proba_baseline[shape_idx]
            p_correct = proba_correct[shape_idx]
            p_wrong_avg = np.mean([wp[shape_idx] for wp in wrong_probas])

            correct_wins = p_correct > p_wrong_avg
            if correct_wins:
                correct_shape_wins += 1
            total += 1

            detail_rows.append({
                'true_shape': true_shape,
                'text': text[:60],
                'p_baseline': float(p_baseline),
                'p_correct_steer': float(p_correct),
                'p_wrong_steer_avg': float(p_wrong_avg),
                'correct_wins': bool(correct_wins),
                'pred_baseline': pred_baseline,
                'pred_correct': pred_correct,
            })

        win_rate = correct_shape_wins / total
        print(f'  Correct-shape steering wins: {correct_shape_wins}/{total} = {win_rate:.1%}')
        print(f'  (Chance = {1/len(SHAPES):.1%})')

        # Per-shape breakdown
        for shape in SHAPES:
            shape_rows = [r for r in detail_rows if r['true_shape'] == shape]
            shape_wins = sum(1 for r in shape_rows if r['correct_wins'])
            avg_p_correct = np.mean([r['p_correct_steer'] for r in shape_rows])
            avg_p_wrong = np.mean([r['p_wrong_steer_avg'] for r in shape_rows])
            avg_p_base = np.mean([r['p_baseline'] for r in shape_rows])
            print(f'    {shape:15s}: wins={shape_wins}/{len(shape_rows)}, '
                  f'P(correct|steer)={avg_p_correct:.3f}, '
                  f'P(correct|wrong)={avg_p_wrong:.3f}, '
                  f'P(correct|none)={avg_p_base:.3f}')

        results_by_target[target_domain] = {
            'win_rate': float(win_rate),
            'total': total,
            'correct_wins': correct_shape_wins,
            'details': detail_rows,
        }

        all_correct += correct_shape_wins
        all_total += total

    # Overall summary
    print('\n' + '=' * 60)
    print('STEERING TRANSFER SUMMARY')
    print('=' * 60)
    overall_win_rate = all_correct / all_total
    print(f'\n  Overall correct-shape steering wins: {all_correct}/{all_total} = {overall_win_rate:.1%}')
    print(f'  Chance: {1/len(SHAPES):.1%} ({100/len(SHAPES):.0f}%)')
    print(f'  Lift over chance: {overall_win_rate - 1/len(SHAPES):+.1%}')

    for domain in DOMAINS:
        r = results_by_target[domain]
        print(f'  {domain:8s}: {r["win_rate"]:.1%} ({r["correct_wins"]}/{r["total"]})')

    if overall_win_rate > 0.6:
        print(f'\n  --> STRONG TRANSFER: Structural steering works across domains')
    elif overall_win_rate > 0.4:
        print(f'\n  --> MODERATE TRANSFER: Structural signal transfers partially')
    elif overall_win_rate > 0.3:
        print(f'\n  --> WEAK TRANSFER: Above chance but noisy')
    else:
        print(f'\n  --> NO TRANSFER: Steering does not transfer structural signal')

    # Binomial test against chance
    from scipy import stats
    binom_p = stats.binom_test(all_correct, all_total, 1/len(SHAPES), alternative='greater')
    print(f'\n  Binomial test vs chance: p = {binom_p:.2e}')
    if binom_p < 0.001:
        print(f'  Highly significant (p < 0.001)')
    elif binom_p < 0.05:
        print(f'  Significant (p < 0.05)')
    else:
        print(f'  Not significant')

    # Save results
    out_dir = Path('results/paper4')
    with open(out_dir / 'steering_transfer_results.json', 'w') as f:
        json.dump({
            'alpha': alpha,
            'steering_layer': steering_layer,
            'n_layers': n_layers,
            'overall_win_rate': float(overall_win_rate),
            'overall_correct': all_correct,
            'overall_total': all_total,
            'chance': float(1/len(SHAPES)),
            'binom_p': float(binom_p),
            'by_domain': {
                d: {'win_rate': r['win_rate'], 'correct': r['correct_wins'], 'total': r['total']}
                for d, r in results_by_target.items()
            },
            'full_details': {d: r['details'] for d, r in results_by_target.items()},
        }, f, indent=2)
    print(f'\n  Results saved to {out_dir}/steering_transfer_results.json')


if __name__ == '__main__':
    main()
