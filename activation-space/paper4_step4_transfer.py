"""Paper 4 Step 4: Transfer test via activation arithmetic.

Instead of running 800 forward passes through Qwen 3B on CPU, we test the
transfer hypothesis directly on saved activation vectors using leave-one-
domain-out cross-validation.

The test: Given a probe from held-out domain D with shape S,
  1. Compute its domain-erased activation (project out 13 domain directions)
  2. Also compute structural prototypes from the OTHER 3 domains
  3. Check: is the held-out probe's structural signature closest to the
     correct-shape prototype (nearest neighbor transfer)?

If yes: structural shape transfers across domains.

We also test a stronger claim: can we RECONSTRUCT a target-domain activation
by taking a source-domain activation's structural component and adding the
target domain's mean signature? (strip A's domain, add B's domain = valid B)

Usage: KMP_DUPLICATE_LIB_OK=TRUE python3 paper4_step4_transfer.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine as cosine_dist
from scipy import stats

from paper4_probes import SHAPES, DOMAINS


def load_data():
    out_dir = Path('results/paper4')
    activations = np.load(out_dir / 'shape_activations.npy')
    domain_dirs = np.load(out_dir / 'removed_domain_directions.npy')
    with open(out_dir / 'shape_labels.json') as f:
        data = json.load(f)
    probes = data['probes']
    shape_labels = np.array([p[0] for p in probes])
    domain_labels = np.array([p[1] for p in probes])
    return activations, domain_dirs, shape_labels, domain_labels


def domain_erase(X, domain_dirs):
    X = X.copy()
    for d in domain_dirs:
        d = d / (np.linalg.norm(d) + 1e-10)
        X = X - np.outer(X @ d, d)
    return X


# ============================================================
# Test 1: Leave-one-domain-out shape classification
# ============================================================

def test_cross_domain_shape_transfer(X_erased, shape_labels, domain_labels):
    """Train shape classifier on 3 domains, test on held-out 4th.
    This is the fundamental transfer test: does structural knowledge
    learned from domains A, B, C predict shape in domain D?"""

    print('\n' + '=' * 60)
    print('TEST 1: CROSS-DOMAIN SHAPE TRANSFER')
    print('(Train on 3 domains, predict shape in held-out domain)')
    print('=' * 60)

    results = {}
    all_preds = []
    all_true = []

    for target in DOMAINS:
        train_mask = domain_labels != target
        test_mask = domain_labels == target

        X_train, y_train = X_erased[train_mask], shape_labels[train_mask]
        X_test, y_test = X_erased[test_mask], shape_labels[test_mask]

        # Logistic regression
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        preds = clf.predict(X_test)
        all_preds.extend(preds)
        all_true.extend(y_test)

        # Per-shape accuracy
        per_shape = {}
        for shape in SHAPES:
            mask = y_test == shape
            if mask.sum() > 0:
                per_shape[shape] = (preds[mask] == y_test[mask]).mean()

        results[target] = {'accuracy': float(acc), 'per_shape': per_shape}
        print(f'\n  Target: {target}')
        print(f'    Accuracy: {acc:.3f} (chance = {1/len(SHAPES):.3f})')
        for shape, sa in per_shape.items():
            print(f'      {shape:15s}: {sa:.3f}')

    overall = np.mean([r['accuracy'] for r in results.values()])
    print(f'\n  Overall cross-domain shape accuracy: {overall:.3f}')

    # Confusion matrix
    print(f'\n  Confusion matrix (rows=true, cols=predicted):')
    cm = confusion_matrix(all_true, all_preds, labels=SHAPES)
    print(f'  {"":15s}', '  '.join(f'{s:>12s}' for s in SHAPES))
    for i, shape in enumerate(SHAPES):
        print(f'  {shape:15s}', '  '.join(f'{cm[i,j]:>12d}' for j in range(len(SHAPES))))

    return results, overall


# ============================================================
# Test 2: Nearest-prototype shape classification
# ============================================================

def test_nearest_prototype_transfer(X_erased, shape_labels, domain_labels):
    """For each held-out probe, find nearest shape prototype from other domains."""

    print('\n' + '=' * 60)
    print('TEST 2: NEAREST STRUCTURAL PROTOTYPE')
    print('(Match held-out probe to shape centroids from other domains)')
    print('=' * 60)

    results = {}

    for target in DOMAINS:
        correct = 0
        total = 0

        for true_shape in SHAPES:
            # Target probes
            target_mask = (domain_labels == target) & (shape_labels == true_shape)
            target_vecs = X_erased[target_mask]

            # Source prototypes (from other 3 domains)
            source_mask = domain_labels != target
            prototypes = {}
            for shape in SHAPES:
                smask = source_mask & (shape_labels == shape)
                prototypes[shape] = X_erased[smask].mean(axis=0)

            # Classify each target probe by nearest prototype
            for vec in target_vecs:
                dists = {s: cosine_dist(vec, p) for s, p in prototypes.items()}
                predicted = min(dists, key=dists.get)
                if predicted == true_shape:
                    correct += 1
                total += 1

        acc = correct / total
        results[target] = float(acc)
        print(f'  {target:8s}: {acc:.3f} ({correct}/{total})')

    overall = np.mean(list(results.values()))
    print(f'\n  Overall nearest-prototype accuracy: {overall:.3f} (chance = {1/len(SHAPES):.3f})')

    return results, overall


# ============================================================
# Test 3: Activation arithmetic transfer
# ============================================================

def test_activation_arithmetic(activations, X_erased, domain_dirs,
                                shape_labels, domain_labels):
    """Test: strip domain A, add domain B = valid activation in B?

    For each (shape, source_domain, target_domain) triple:
      1. Take source probe activation
      2. Erase source domain signal
      3. Add target domain's mean activation (rehydrate)
      4. Check: is the result close to actual target-domain same-shape probes?
    """

    print('\n' + '=' * 60)
    print('TEST 3: ACTIVATION ARITHMETIC (STRIP + REHYDRATE)')
    print('(Strip domain A, add domain B signature = valid B activation?)')
    print('=' * 60)

    # Compute domain-specific components: domain_mean - global_mean
    global_mean = activations.mean(axis=0)
    domain_means = {}
    for domain in DOMAINS:
        mask = domain_labels == domain
        domain_means[domain] = activations[mask].mean(axis=0)

    # For each transfer pair
    transfer_results = []

    for source_domain in DOMAINS:
        for target_domain in DOMAINS:
            if source_domain == target_domain:
                continue

            for shape in SHAPES:
                # Source probes
                source_mask = (domain_labels == source_domain) & (shape_labels == shape)
                source_acts = activations[source_mask]

                # Target probes (ground truth)
                target_mask = (domain_labels == target_domain) & (shape_labels == shape)
                target_acts = activations[target_mask]
                target_mean = target_acts.mean(axis=0)

                # Wrong-shape target probes (control)
                wrong_shapes = [s for s in SHAPES if s != shape]

                for src_vec in source_acts:
                    # Strip source domain, add target domain
                    erased = domain_erase(src_vec.reshape(1, -1), domain_dirs).squeeze()
                    rehydrated = erased + (domain_means[target_domain] - global_mean)

                    # Distance to actual target same-shape
                    dist_correct = cosine_dist(rehydrated, target_mean)

                    # Distance to actual target wrong-shape (average)
                    dists_wrong = []
                    for ws in wrong_shapes:
                        ws_mask = (domain_labels == target_domain) & (shape_labels == ws)
                        ws_mean = activations[ws_mask].mean(axis=0)
                        dists_wrong.append(cosine_dist(rehydrated, ws_mean))
                    dist_wrong_avg = np.mean(dists_wrong)

                    transfer_results.append({
                        'source_domain': source_domain,
                        'target_domain': target_domain,
                        'shape': shape,
                        'dist_correct': float(dist_correct),
                        'dist_wrong_avg': float(dist_wrong_avg),
                        'correct_closer': dist_correct < dist_wrong_avg,
                    })

    # Aggregate
    correct_closer = sum(1 for r in transfer_results if r['correct_closer'])
    total = len(transfer_results)
    overall_rate = correct_closer / total

    print(f'\n  Rehydrated activation closer to correct shape: {correct_closer}/{total} = {overall_rate:.1%}')
    print(f'  Chance: ~50% (binary: correct vs wrong-shape average)')

    # By transfer direction
    print(f'\n  By transfer direction:')
    for source in DOMAINS:
        for target in DOMAINS:
            if source == target:
                continue
            pair_results = [r for r in transfer_results if
                          r['source_domain'] == source and r['target_domain'] == target]
            pair_rate = sum(1 for r in pair_results if r['correct_closer']) / len(pair_results)
            print(f'    {source:8s} -> {target:8s}: {pair_rate:.1%} ({sum(1 for r in pair_results if r["correct_closer"])}/{len(pair_results)})')

    # By shape
    print(f'\n  By shape:')
    for shape in SHAPES:
        shape_results = [r for r in transfer_results if r['shape'] == shape]
        shape_rate = sum(1 for r in shape_results if r['correct_closer']) / len(shape_results)
        avg_dist_correct = np.mean([r['dist_correct'] for r in shape_results])
        avg_dist_wrong = np.mean([r['dist_wrong_avg'] for r in shape_results])
        print(f'    {shape:15s}: {shape_rate:.1%}, dist_correct={avg_dist_correct:.4f}, dist_wrong={avg_dist_wrong:.4f}')

    # Paired t-test: is dist_correct < dist_wrong systematically?
    dists_c = [r['dist_correct'] for r in transfer_results]
    dists_w = [r['dist_wrong_avg'] for r in transfer_results]
    t_stat, p_val = stats.ttest_rel(dists_c, dists_w)
    print(f'\n  Paired t-test (correct < wrong): t={t_stat:.3f}, p={p_val:.2e}')
    effect_size = (np.mean(dists_w) - np.mean(dists_c)) / np.std(np.array(dists_w) - np.array(dists_c))
    print(f'  Effect size (Cohen d): {effect_size:.3f}')

    # Binomial test
    binom_p = stats.binomtest(correct_closer, total, 0.5, alternative='greater').pvalue
    print(f'  Binomial test vs 50%: p={binom_p:.2e}')

    return transfer_results, overall_rate


# ============================================================
# Test 4: Transfer quality vs structural similarity
# ============================================================

def test_transfer_vs_similarity(X_erased, shape_labels, domain_labels):
    """Falsification F1: transfer accuracy should correlate with structural similarity."""

    print('\n' + '=' * 60)
    print('TEST 4: TRANSFER QUALITY vs STRUCTURAL SIMILARITY')
    print('(F1: closer structures should transfer better)')
    print('=' * 60)

    # Compute inter-shape distances in the structural subspace
    shape_centroids = {}
    for shape in SHAPES:
        mask = shape_labels == shape
        shape_centroids[shape] = X_erased[mask].mean(axis=0)

    print(f'\n  Inter-shape cosine distances:')
    for i, s1 in enumerate(SHAPES):
        for s2 in SHAPES[i+1:]:
            d = cosine_dist(shape_centroids[s1], shape_centroids[s2])
            print(f'    {s1:15s} <-> {s2:15s}: {d:.4f}')

    # Per-domain shape discriminability
    print(f'\n  Per-domain shape classification (within-domain):')
    for domain in DOMAINS:
        mask = domain_labels == domain
        X_dom = X_erased[mask]
        y_dom = shape_labels[mask]
        clf = LogisticRegression(max_iter=2000, solver='lbfgs')
        scores = cross_val_score(clf, X_dom, y_dom, cv=5, scoring='accuracy')
        print(f'    {domain:8s}: {scores.mean():.3f} +/- {scores.std():.3f}')


# ============================================================
# Main
# ============================================================

def main():
    print('Paper 4 Step 4: Transfer Tests')
    print('=' * 60)

    activations, domain_dirs, shape_labels, domain_labels = load_data()
    X_erased = domain_erase(activations, domain_dirs)
    print(f'Loaded: {activations.shape[0]} probes, {activations.shape[1]}d')
    print(f'Domain directions: {len(domain_dirs)}')

    # Run all tests
    t1_results, t1_overall = test_cross_domain_shape_transfer(X_erased, shape_labels, domain_labels)
    t2_results, t2_overall = test_nearest_prototype_transfer(X_erased, shape_labels, domain_labels)
    t3_results, t3_overall = test_activation_arithmetic(activations, X_erased, domain_dirs, shape_labels, domain_labels)
    test_transfer_vs_similarity(X_erased, shape_labels, domain_labels)

    # Final summary
    print('\n' + '=' * 60)
    print('TRANSFER TEST SUMMARY')
    print('=' * 60)
    print(f'\n  Test 1 (cross-domain shape classification): {t1_overall:.3f} (chance = 0.250)')
    print(f'  Test 2 (nearest prototype):                 {t2_overall:.3f} (chance = 0.250)')
    print(f'  Test 3 (strip + rehydrate):                 {t3_overall:.1%} (chance = 50%)')

    # Overall assessment
    if t1_overall > 0.7 and t3_overall > 0.7:
        verdict = "STRONG: Structural transfer works across domains"
    elif t1_overall > 0.5 and t3_overall > 0.6:
        verdict = "MODERATE: Partial structural transfer"
    elif t1_overall > 0.35:
        verdict = "WEAK: Above chance but noisy"
    else:
        verdict = "NEGATIVE: No structural transfer detected"

    print(f'\n  Verdict: {verdict}')

    # Save
    out_dir = Path('results/paper4')
    with open(out_dir / 'transfer_test_results.json', 'w') as f:
        json.dump({
            'test1_cross_domain_accuracy': float(t1_overall),
            'test1_by_domain': {d: r['accuracy'] for d, r in t1_results.items()},
            'test2_nearest_prototype_accuracy': float(t2_overall),
            'test2_by_domain': t2_results,
            'test3_strip_rehydrate_rate': float(t3_overall),
            'test3_n_transfers': len(t3_results),
        }, f, indent=2)
    print(f'\n  Results saved to {out_dir}/transfer_test_results.json')


if __name__ == '__main__':
    main()
