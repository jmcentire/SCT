#!/usr/bin/env python3
"""Paper 5: Capability Manifold Surveillance (Qwen 2.5 1.5B).

Detects model distillation attacks via activation fingerprint topology.
Seven experimental passes:

  Pass 1: Fingerprint extraction for all probes
  Pass 2: Legitimate session baseline characterization
  Pass 3: Attack session simulation and detection
  Pass 4: ROC analysis across all three detectors
  Pass 5: Adversarial efficiency bound computation
  Pass 6: Window size sensitivity analysis
  Pass 7: PoH scoring and signal decomposition

Usage:
  # Run all passes (requires GPU for pass 1):
  python paper5_cms_qwen.py --run-all --device cuda

  # Run individual passes (passes 2-7 work on CPU with cached fingerprints):
  python paper5_cms_qwen.py --pass 1 --device cuda
  python paper5_cms_qwen.py --pass 2 --device cpu
  python paper5_cms_qwen.py --pass 3 --device cpu
"""
import os, sys, time, json, gc, argparse
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
torch.set_num_threads(1)
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) or '.')

from src.fingerprint.capture import ActivationCapture, capture_at_checkpoint
from src.regime.detect import cosine_similarity_batch
from src.cms.sessions import (
    Session, generate_all_sessions, compute_coverage,
    generate_legitimate_single_domain, generate_legitimate_mixed_domain,
    generate_legitimate_power_user, generate_naive_attack,
    generate_systematic_attack, generate_adaptive_attack,
)
from src.cms.detectors import (
    PRADADetector, VarDetectDetector, CMSDetector,
    evaluate_session, session_score,
)
from src.cms.scoring import ExtractionScorer, MultiSignalScorer, ScoringState
from src.cms.metrics import (
    roc_curve, operating_point, compare_detectors,
    per_attacker_roc, summary_table,
)
from src.cms.adversarial import coverage_efficiency_sweep, attacker_cost_model
from src.analysis.plot_cms import generate_all_figures

# ============================================================
# Domain Probes (reused from Paper 3)
# ============================================================

# Import directly from paper3 to avoid duplication
from paper3_constellation import DOMAIN_PROBES

# ============================================================
# Constants
# ============================================================

RESULTS_DIR = Path('results/paper5_qwen')

# Session parameters
WINDOW_SIZES = [10, 15, 20, 25, 30, 40, 50, 75, 100]
DEFAULT_WINDOW = 50

# Detection thresholds for adversarial analysis
TAU_MU_VALUES = [0.0, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# ============================================================
# Helpers
# ============================================================

def load_model(device='cpu'):
    """Load Qwen 2.5 1.5B pretrained model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'Qwen/Qwen2.5-1.5B'
    print(f'Loading model {model_name}...', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'  Tokenizer loaded', flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f'  Model loaded: {param_count / 1e9:.1f}B parameters', flush=True)

    return model, tokenizer


def tokenize_probes(domain_probes, tokenizer, max_length=128):
    """Tokenize all domain probes into a single batch."""
    all_texts = []
    probe_domains = []
    for domain, probes in domain_probes.items():
        for probe in probes:
            all_texts.append(probe)
            probe_domains.append(domain)

    encoded = tokenizer(
        all_texts, padding=True, truncation=True,
        max_length=max_length, return_tensors='pt',
    )

    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'texts': all_texts,
        'domains': probe_domains,
    }


def save_results(results, filename):
    """Save results dict as JSON (numpy-safe)."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if hasattr(obj, '__dataclass_fields__'):
            return convert(obj.__dict__)
        return obj

    filepath = RESULTS_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2, default=str)
    print(f'  Saved: {filepath}')


def load_cached(filename):
    """Load cached results if available."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath) as f:
            return json.load(f)
    return None


# ============================================================
# Pass 1: Fingerprint Extraction
# ============================================================

def run_pass1(args):
    """Extract activation fingerprints for all domain probes."""
    print('\n' + '='*60)
    print('PASS 1: Fingerprint Extraction')
    print('='*60)

    model, tokenizer = load_model(args.device)
    probe_set = tokenize_probes(DOMAIN_PROBES, tokenizer)

    print(f'  Total probes: {len(probe_set["texts"])}')
    print(f'  Domains: {list(DOMAIN_PROBES.keys())}')
    print(f'  Probes per domain: {[len(v) for v in DOMAIN_PROBES.values()]}')

    # Capture activations at final hidden layer
    t0 = time.time()
    activations = capture_at_checkpoint(
        model, probe_set, ['final_hidden'],
        device=args.device, batch_size=32,
    )
    elapsed = time.time() - t0

    fingerprints = activations['final_hidden']  # (n_probes, hidden_dim)
    print(f'  Fingerprint shape: {fingerprints.shape}')
    print(f'  Extraction time: {elapsed:.1f}s')

    # Also extract input embeddings for PRADA baseline
    # (token embedding layer, not hidden states)
    with torch.no_grad():
        input_ids = probe_set['input_ids'].to(args.device)
        # Get token embeddings (before transformer layers)
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            wte = model.model.embed_tokens(input_ids)  # (n, seq, dim)
        else:
            wte = model.get_input_embeddings()(input_ids)
        input_embeddings = wte.mean(dim=1).float().cpu().numpy()  # (n, dim)

    print(f'  Input embedding shape: {input_embeddings.shape}')

    # Compute baseline statistics
    norms = np.linalg.norm(fingerprints, axis=1)
    print(f'  Fingerprint norms: mean={norms.mean():.3f}, std={norms.std():.3f}')

    # Per-domain similarity baselines
    offsets = {}
    idx = 0
    for domain in DOMAIN_PROBES:
        offsets[domain] = idx
        idx += len(DOMAIN_PROBES[domain])

    domain_stats = {}
    for domain in DOMAIN_PROBES:
        start = offsets[domain]
        end = start + len(DOMAIN_PROBES[domain])
        domain_fps = fingerprints[start:end]

        # Within-domain pairwise cosine
        n = domain_fps.shape[0]
        normed = domain_fps / (np.linalg.norm(domain_fps, axis=1, keepdims=True) + 1e-8)
        sim_matrix = normed @ normed.T
        mask = np.triu_indices(n, k=1)
        within_sims = sim_matrix[mask]

        # Centered cosine (Pearson)
        centered = domain_fps - domain_fps.mean(axis=1, keepdims=True)
        c_normed = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8)
        c_sim_matrix = c_normed @ c_normed.T
        centered_sims = c_sim_matrix[mask]

        domain_stats[domain] = {
            'n_probes': n,
            'cosine_mean': float(within_sims.mean()),
            'cosine_std': float(within_sims.std()),
            'centered_cosine_mean': float(centered_sims.mean()),
            'centered_cosine_std': float(centered_sims.std()),
        }
        print(f'  {domain:10s}: cos={within_sims.mean():.4f}±{within_sims.std():.4f}  '
              f'pearson={centered_sims.mean():.4f}±{centered_sims.std():.4f}')

    # Cross-domain similarity
    all_normed = fingerprints / (np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8)
    cross_domain_sims = []
    for d1 in DOMAIN_PROBES:
        for d2 in DOMAIN_PROBES:
            if d1 >= d2:
                continue
            s1, e1 = offsets[d1], offsets[d1] + len(DOMAIN_PROBES[d1])
            s2, e2 = offsets[d2], offsets[d2] + len(DOMAIN_PROBES[d2])
            sims = all_normed[s1:e1] @ all_normed[s2:e2].T
            cross_domain_sims.append(sims.mean())

    print(f'  Cross-domain cosine: mean={np.mean(cross_domain_sims):.4f}')

    results = {
        'fingerprints_shape': list(fingerprints.shape),
        'input_embeddings_shape': list(input_embeddings.shape),
        'domain_stats': domain_stats,
        'cross_domain_cosine_mean': float(np.mean(cross_domain_sims)),
        'extraction_time_s': elapsed,
        'model': 'Qwen/Qwen2.5-1.5B',
        'n_probes': len(probe_set['texts']),
        'domains': list(DOMAIN_PROBES.keys()),
        'probe_domains': probe_set['domains'],
    }

    # Save fingerprints as numpy
    np.save(RESULTS_DIR / 'fingerprints.npy', fingerprints)
    np.save(RESULTS_DIR / 'input_embeddings.npy', input_embeddings)
    save_results(results, 'pass1_fingerprints.json')

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fingerprints, input_embeddings, results


# ============================================================
# Pass 2: Legitimate Session Baseline
# ============================================================

def run_pass2(args, fingerprints=None, input_embeddings=None):
    """Characterize similarity distributions for legitimate sessions."""
    print('\n' + '='*60)
    print('PASS 2: Legitimate Session Baseline')
    print('='*60)

    if fingerprints is None:
        fingerprints = np.load(RESULTS_DIR / 'fingerprints.npy')
    if input_embeddings is None:
        input_embeddings = np.load(RESULTS_DIR / 'input_embeddings.npy')

    print(f'  Fingerprints: {fingerprints.shape}')

    # Generate legitimate sessions
    rng = np.random.RandomState(42)
    sessions = {
        'single_domain': generate_legitimate_single_domain(
            DOMAIN_PROBES, k=DEFAULT_WINDOW, n_sessions=250,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
        'mixed_domain': generate_legitimate_mixed_domain(
            DOMAIN_PROBES, k=DEFAULT_WINDOW, n_sessions=250,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
        'power_user': generate_legitimate_power_user(
            DOMAIN_PROBES, k=DEFAULT_WINDOW, n_sessions=100,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
    }

    # Compute similarity distributions for each session type
    from src.cms.detectors import _pairwise_centered_cosine, _pairwise_cosine

    similarity_distributions = {}
    session_stats = {}

    for stype, sess_list in sessions.items():
        sims = []
        raw_sims = []
        entropies = []
        coverages = []

        for session in sess_list:
            fps = fingerprints[session.probe_indices]

            # Centered cosine (primary metric)
            pairwise = _pairwise_centered_cosine(fps)
            sims.append(float(pairwise.mean()))

            # Raw cosine for comparison
            raw = _pairwise_cosine(fps)
            raw_sims.append(float(raw.mean()))

            # Entropy
            from src.cms.detectors import _similarity_entropy
            entropies.append(_similarity_entropy(pairwise))

            # Coverage
            coverages.append(compute_coverage(session, fingerprints))

        similarity_distributions[stype] = sims

        session_stats[stype] = {
            'n_sessions': len(sess_list),
            'centered_cosine_mean': float(np.mean(sims)),
            'centered_cosine_std': float(np.std(sims)),
            'raw_cosine_mean': float(np.mean(raw_sims)),
            'raw_cosine_std': float(np.std(raw_sims)),
            'entropy_mean': float(np.mean(entropies)),
            'entropy_std': float(np.std(entropies)),
            'coverage_mean': float(np.mean(coverages)),
            'coverage_std': float(np.std(coverages)),
        }

        print(f'  {stype:15s}: μ_centered={np.mean(sims):.4f}±{np.std(sims):.4f}  '
              f'entropy={np.mean(entropies):.3f}  coverage={np.mean(coverages):.3f}')

    results = {
        'session_stats': session_stats,
        'similarity_distributions': similarity_distributions,
        'window_size': DEFAULT_WINDOW,
    }

    save_results(results, 'pass2_baseline.json')
    return sessions, results


# ============================================================
# Pass 3: Attack Simulation and Detection
# ============================================================

def run_pass3(args, fingerprints=None, input_embeddings=None):
    """Simulate attacks and run all three detectors."""
    print('\n' + '='*60)
    print('PASS 3: Attack Simulation and Detection')
    print('='*60)

    if fingerprints is None:
        fingerprints = np.load(RESULTS_DIR / 'fingerprints.npy')
    if input_embeddings is None:
        input_embeddings = np.load(RESULTS_DIR / 'input_embeddings.npy')

    rng = np.random.RandomState(42)

    # Generate all sessions (legitimate + attack)
    print('  Generating sessions...')
    all_sessions = generate_all_sessions(
        DOMAIN_PROBES, fingerprints,
        k=DEFAULT_WINDOW, seed=42,
        n_legit_single=250, n_legit_mixed=250, n_legit_power=100,
        n_attack_naive=200, n_attack_systematic=200, n_attack_adaptive=100,
    )

    total_legit = sum(len(v) for k, v in all_sessions.items() if v[0].label == 0)
    total_attack = sum(len(v) for k, v in all_sessions.items() if v[0].label == 1)
    print(f'  Legitimate sessions: {total_legit}')
    print(f'  Attack sessions: {total_attack}')

    # Initialize detectors
    detectors = {
        'prada': PRADADetector(),
        'vardetect': VarDetectDetector(projection_dim=64),
        'cms': CMSDetector(use_centered_cosine=True),
    }

    # Run each detector on all sessions
    all_scores = {}  # detector -> session_type -> list of scores
    all_detections = {}  # detector -> session_type -> list of list of DetectionResults

    for det_name, detector in detectors.items():
        print(f'\n  Running {det_name.upper()} detector...')
        all_scores[det_name] = {}
        all_detections[det_name] = {}

        for stype, sess_list in all_sessions.items():
            scores = []
            detections_list = []

            for session in sess_list:
                # Choose input based on detector type
                if det_name == 'prada':
                    fp = input_embeddings  # PRADA uses input space
                else:
                    fp = fingerprints  # VarDetect and CMS use activation space

                results = evaluate_session(detector, fp, session.probe_indices,
                                           window_size=DEFAULT_WINDOW)
                score = session_score(results, aggregation='max')
                scores.append(score)
                detections_list.append(results)

            all_scores[det_name][stype] = scores
            all_detections[det_name][stype] = detections_list
            print(f'    {stype:20s}: mean_score={np.mean(scores):.4f}±{np.std(scores):.4f}')

    # Compute similarity distributions for attack sessions (for Figure 1)
    from src.cms.detectors import _pairwise_centered_cosine
    similarity_distributions = {}
    for stype, sess_list in all_sessions.items():
        sims = []
        for session in sess_list:
            fps = fingerprints[session.probe_indices]
            pairwise = _pairwise_centered_cosine(fps)
            sims.append(float(pairwise.mean()))
        similarity_distributions[stype] = sims

    results = {
        'scores': all_scores,
        'similarity_distributions': similarity_distributions,
        'session_counts': {k: len(v) for k, v in all_sessions.items()},
        'window_size': DEFAULT_WINDOW,
    }

    save_results(results, 'pass3_detection.json')
    return all_sessions, all_scores, all_detections, results


# ============================================================
# Pass 4: ROC Analysis
# ============================================================

def run_pass4(args, all_scores=None):
    """Compute ROC curves across detectors and attacker types."""
    print('\n' + '='*60)
    print('PASS 4: ROC Analysis')
    print('='*60)

    if all_scores is None:
        cached = load_cached('pass3_detection.json')
        if cached is None:
            print('  ERROR: Pass 3 results required. Run pass 3 first.')
            return None
        all_scores = cached['scores']

    legit_types = ['single_domain', 'mixed_domain', 'power_user']
    attack_types = ['naive_attack', 'systematic_attack', 'adaptive_attack']

    roc_results = {}  # detector -> attacker_type -> ROCResult
    comparison_results = {}

    for det_name in all_scores:
        print(f'\n  {det_name.upper()}:')
        roc_results[det_name] = {}

        # Combine all legitimate scores
        legit_scores = []
        for lt in legit_types:
            if lt in all_scores[det_name]:
                legit_scores.extend(all_scores[det_name][lt])
        legit_scores = np.array(legit_scores)

        # Per-attacker ROC
        for at in attack_types:
            if at not in all_scores[det_name]:
                continue
            attack_scores = np.array(all_scores[det_name][at])

            scores = np.concatenate([legit_scores, attack_scores])
            labels = np.concatenate([
                np.zeros(len(legit_scores)),
                np.ones(len(attack_scores)),
            ])

            roc = roc_curve(scores, labels)
            roc_results[det_name][at] = roc

            op_1 = operating_point(roc, 0.01)
            op_5 = operating_point(roc, 0.05)

            print(f'    vs {at:20s}: AUC={roc.auc:.4f}  '
                  f'TPR@1%FPR={op_1["tpr"]:.4f}  '
                  f'TPR@5%FPR={op_5["tpr"]:.4f}')

        # All attacks combined
        all_attack_scores = []
        for at in attack_types:
            if at in all_scores[det_name]:
                all_attack_scores.extend(all_scores[det_name][at])
        all_attack_scores = np.array(all_attack_scores)

        combined_scores = np.concatenate([legit_scores, all_attack_scores])
        combined_labels = np.concatenate([
            np.zeros(len(legit_scores)),
            np.ones(len(all_attack_scores)),
        ])
        combined_roc = roc_curve(combined_scores, combined_labels)
        roc_results[det_name]['all_attacks'] = combined_roc
        op_1 = operating_point(combined_roc, 0.01)
        print(f'    vs ALL ATTACKS     : AUC={combined_roc.auc:.4f}  '
              f'TPR@1%FPR={op_1["tpr"]:.4f}')

    # Comparison table
    print('\n  === Detector Comparison (All Attacks Combined) ===')
    for det_name in roc_results:
        roc = roc_results[det_name].get('all_attacks')
        if roc:
            op_1 = operating_point(roc, 0.01)
            op_5 = operating_point(roc, 0.05)
            print(f'  {det_name:10s}: AUC={roc.auc:.4f}  '
                  f'TPR@1%FPR={op_1["tpr"]:.4f}  '
                  f'TPR@5%FPR={op_5["tpr"]:.4f}')

    # Serialize ROC results
    serializable = {}
    for det_name, attacker_rocs in roc_results.items():
        serializable[det_name] = {}
        for at, roc in attacker_rocs.items():
            serializable[det_name][at] = {
                'auc': roc.auc,
                'fpr': roc.fpr.tolist(),
                'tpr': roc.tpr.tolist(),
                'operating_points': {
                    'fpr_0.01': operating_point(roc, 0.01),
                    'fpr_0.05': operating_point(roc, 0.05),
                    'fpr_0.10': operating_point(roc, 0.10),
                },
            }

    save_results(serializable, 'pass4_roc.json')
    return roc_results


# ============================================================
# Pass 5: Adversarial Efficiency Bound
# ============================================================

def run_pass5(args, fingerprints=None):
    """Compute coverage-efficiency curve under detection constraints."""
    print('\n' + '='*60)
    print('PASS 5: Adversarial Efficiency Bound')
    print('='*60)

    if fingerprints is None:
        fingerprints = np.load(RESULTS_DIR / 'fingerprints.npy')

    print(f'  Sweeping τ_μ values: {TAU_MU_VALUES}')
    print(f'  Window size: {DEFAULT_WINDOW}')

    sweep = coverage_efficiency_sweep(
        fingerprints,
        tau_values=TAU_MU_VALUES,
        k=DEFAULT_WINDOW,
        n_trials=50,
        seed=42,
    )

    print(f'\n  Baseline (unconstrained) coverage: {sweep["baseline_coverage"]:.4f}')
    print(f'\n  {"τ_μ":>6s}  {"Coverage":>10s}  {"Efficiency":>12s}  {"Ratio":>8s}')
    print(f'  {"-"*6}  {"-"*10}  {"-"*12}  {"-"*8}')
    for tau, cov, ratio in zip(sweep['tau_values'], sweep['coverages'],
                                sweep['efficiency_ratios']):
        print(f'  {tau:6.2f}  {cov:10.4f}  {cov/DEFAULT_WINDOW:12.6f}  {ratio:8.4f}')

    # Cost model
    costs = attacker_cost_model(sweep, target_coverage=0.8)
    print(f'\n  Cost to achieve 80% of baseline coverage:')
    for tau, cost_data in costs.items():
        mult = cost_data['cost_multiplier']
        if mult < 100:
            print(f'    τ_μ={tau:.2f}: {mult:.1f}x sessions needed')
        else:
            print(f'    τ_μ={tau:.2f}: effectively infeasible')

    results = {
        'sweep': {
            'tau_values': sweep['tau_values'],
            'coverages': sweep['coverages'],
            'efficiencies': sweep['efficiencies'],
            'baseline_coverage': sweep['baseline_coverage'],
            'efficiency_ratios': sweep['efficiency_ratios'],
        },
        'cost_model': costs,
        'window_size': DEFAULT_WINDOW,
    }

    save_results(results, 'pass5_adversarial.json')
    return sweep, results


# ============================================================
# Pass 6: Window Size Sensitivity
# ============================================================

def run_pass6(args, fingerprints=None, input_embeddings=None):
    """Analyze detection performance across window sizes."""
    print('\n' + '='*60)
    print('PASS 6: Window Size Sensitivity Analysis')
    print('='*60)

    if fingerprints is None:
        fingerprints = np.load(RESULTS_DIR / 'fingerprints.npy')
    if input_embeddings is None:
        input_embeddings = np.load(RESULTS_DIR / 'input_embeddings.npy')

    print(f'  Window sizes: {WINDOW_SIZES}')

    # Use CMS detector only for window analysis
    detector = CMSDetector(use_centered_cosine=True)

    results_by_window = {}
    rng = np.random.RandomState(42)

    for ws in WINDOW_SIZES:
        print(f'\n  Window size k={ws}:')

        # Generate sessions at this window size
        sessions = generate_all_sessions(
            DOMAIN_PROBES, fingerprints,
            k=ws, seed=42,
            n_legit_single=100, n_legit_mixed=100, n_legit_power=50,
            n_attack_naive=100, n_attack_systematic=100, n_attack_adaptive=50,
        )

        # Run CMS detector
        legit_scores = []
        attack_scores = []

        for stype, sess_list in sessions.items():
            for session in sess_list:
                det_results = evaluate_session(detector, fingerprints,
                                                session.probe_indices,
                                                window_size=ws)
                score = session_score(det_results, aggregation='max')
                if session.label == 0:
                    legit_scores.append(score)
                else:
                    attack_scores.append(score)

        legit_scores = np.array(legit_scores)
        attack_scores = np.array(attack_scores)

        scores = np.concatenate([legit_scores, attack_scores])
        labels = np.concatenate([np.zeros(len(legit_scores)),
                                  np.ones(len(attack_scores))])

        roc = roc_curve(scores, labels)
        op_1 = operating_point(roc, 0.01)

        results_by_window[ws] = {
            'auc': roc.auc,
            'tpr_at_1pct_fpr': op_1['tpr'],
            'n_legit': len(legit_scores),
            'n_attack': len(attack_scores),
            'legit_score_mean': float(legit_scores.mean()),
            'attack_score_mean': float(attack_scores.mean()),
        }

        print(f'    AUC={roc.auc:.4f}  TPR@1%FPR={op_1["tpr"]:.4f}')

    # Format for plotting
    window_analysis = {
        'window_sizes': WINDOW_SIZES,
        'aucs': [results_by_window[ws]['auc'] for ws in WINDOW_SIZES],
        'tprs_at_1pct_fpr': [results_by_window[ws]['tpr_at_1pct_fpr'] for ws in WINDOW_SIZES],
    }

    results = {
        'by_window': results_by_window,
        'window_analysis': window_analysis,
    }

    save_results(results, 'pass6_windows.json')
    return results


# ============================================================
# Pass 7: PoH Scoring and Signal Decomposition
# ============================================================

def run_pass7(args, all_detections=None, fingerprints=None):
    """Compute PoH scores and signal decomposition."""
    print('\n' + '='*60)
    print('PASS 7: PoH Scoring and Signal Decomposition')
    print('='*60)

    if fingerprints is None:
        fingerprints = np.load(RESULTS_DIR / 'fingerprints.npy')

    # If we don't have cached detections, regenerate
    if all_detections is None:
        print('  Regenerating detections for PoH analysis...')
        detector = CMSDetector(use_centered_cosine=True)
        sessions = generate_all_sessions(
            DOMAIN_PROBES, fingerprints,
            k=DEFAULT_WINDOW, seed=42,
            n_legit_single=100, n_legit_mixed=100, n_legit_power=50,
            n_attack_naive=100, n_attack_systematic=100, n_attack_adaptive=50,
        )
        all_detections = {'cms': {}}
        for stype, sess_list in sessions.items():
            det_list = []
            for session in sess_list:
                results = evaluate_session(detector, fingerprints,
                                            session.probe_indices,
                                            window_size=DEFAULT_WINDOW)
                det_list.append(results)
            all_detections['cms'][stype] = det_list

    # PoH scoring — simulate multi-session accounts.
    # The PoH model accumulates evidence over multiple sessions from the
    # same account. Each "account" processes N_SESSIONS_PER_ACCOUNT sequential
    # sessions, modeling a user (or attacker) making repeated API calls.
    #
    # Calibrate anomaly_threshold empirically: collect all legitimate
    # detection scores, set threshold at p95 to achieve ~5% FPR.
    N_SESSIONS_PER_ACCOUNT = 20
    N_ACCOUNTS = 25

    # Collect legitimate detection scores for calibration.
    # The key insight: legitimate sessions have high variance in scores
    # (bimodal due to code domain), while systematic attacks are consistent.
    # Use p75 as threshold — this means ~25% of legitimate windows trigger,
    # but they don't sustain, while attack windows consistently trigger.
    legit_det_scores = []
    for stype in ['single_domain', 'mixed_domain', 'power_user']:
        if stype in all_detections.get('cms', {}):
            for det_list in all_detections['cms'][stype]:
                for det in det_list:
                    legit_det_scores.append(det.score)

    # The legitimate score distribution is bimodal: most sessions score
    # 0.08-0.30 (same-domain or similar-domain probes), but ~16% score
    # 0.50-0.67 (code-domain probes with very different activation patterns).
    # The threshold sits at the gap between modes (0.35-0.50).
    # At GPT-2 124M scale, this means some false positives from diverse
    # legitimate use — a known limitation that improves at larger scale
    # where activation spaces differentiate more.
    if legit_det_scores:
        # Use the valley between modes: midpoint of p70 and p80
        p70 = float(np.percentile(legit_det_scores, 70))
        p80 = float(np.percentile(legit_det_scores, 80))
        threshold = (p70 + p80) / 2
        print(f'  Calibrated anomaly threshold: {threshold:.4f} (bimodal valley)')
    else:
        threshold = 0.35

    scorer = ExtractionScorer(
        anomaly_threshold=threshold,
        decay_half_life=8,        # Fast decay: isolated anomalies fade quickly
        base_weight=0.08,         # Moderate base weight per anomalous window
        cost_escalation=2.5,      # Consecutive anomalies escalate fast
    )

    poh_scores = {}
    poh_final = {}

    for stype in all_detections.get('cms', {}):
        det_list = all_detections['cms'][stype]
        all_histories = []
        final_scores = []

        for acct in range(min(N_ACCOUNTS, max(1, len(det_list) // N_SESSIONS_PER_ACCOUNT))):
            state = ScoringState()
            history = []
            start = acct * N_SESSIONS_PER_ACCOUNT
            end = min(start + N_SESSIONS_PER_ACCOUNT, len(det_list))

            for sess_idx in range(start, end):
                for det in det_list[sess_idx]:
                    scorer.update(state, det)
                    history.append(state.score)

            all_histories.append(history)
            final_scores.append(state.score)

        # Average trajectory for plotting
        if all_histories:
            max_len = max(len(h) for h in all_histories)
            padded = [h + [h[-1]] * (max_len - len(h)) for h in all_histories]
            avg_history = np.mean(padded, axis=0).tolist()
        else:
            avg_history = []

        poh_scores[stype] = avg_history
        poh_final[stype] = {
            'mean_final_score': float(np.mean(final_scores)) if final_scores else 0,
            'std_final_score': float(np.std(final_scores)) if final_scores else 0,
            'max_final_score': float(np.max(final_scores)) if final_scores else 0,
            'pct_above_0.5': float(np.mean([s > 0.5 for s in final_scores])) if final_scores else 0,
            'pct_above_0.7': float(np.mean([s > 0.7 for s in final_scores])) if final_scores else 0,
        }

        print(f'  {stype:20s}: final_score={np.mean(final_scores):.4f}±{np.std(final_scores):.4f}  '
              f'>{0.5}={poh_final[stype]["pct_above_0.5"]*100:.0f}%  '
              f'>{0.7}={poh_final[stype]["pct_above_0.7"]*100:.0f}%')

    # Signal decomposition: average signal components per session type
    multi_scorer = MultiSignalScorer()
    signal_decomp = {}

    for stype in all_detections.get('cms', {}):
        det_list = all_detections['cms'][stype]
        all_components = {}

        for detections in det_list:
            for det in detections:
                breakdown = multi_scorer.score_detection(det)
                for sig_name, sig_data in breakdown.get('signals', {}).items():
                    if sig_name not in all_components:
                        all_components[sig_name] = []
                    all_components[sig_name].append(sig_data['raw'])

        signal_decomp[stype] = {
            name: float(np.mean(vals)) for name, vals in all_components.items()
        }

    results = {
        'poh_scores': poh_scores,
        'poh_final': poh_final,
        'signal_decomposition': signal_decomp,
    }

    save_results(results, 'pass7_scoring.json')
    return results


# ============================================================
# Figure Generation
# ============================================================

def generate_figures(args):
    """Generate all publication-quality figures from cached results."""
    print('\n' + '='*60)
    print('GENERATING FIGURES')
    print('='*60)

    results = {}

    # Load all cached results
    pass2 = load_cached('pass2_baseline.json')
    pass3 = load_cached('pass3_detection.json')
    pass4 = load_cached('pass4_roc.json')
    pass5 = load_cached('pass5_adversarial.json')
    pass6 = load_cached('pass6_windows.json')
    pass7 = load_cached('pass7_scoring.json')

    # Figure 1: Similarity distributions
    if pass3 and 'similarity_distributions' in pass3:
        results['similarity_distributions'] = pass3['similarity_distributions']

    # Figure 2: ROC curves
    if pass4:
        from src.cms.metrics import ROCResult
        roc_by_detector = {}
        for det_name, attacker_rocs in pass4.items():
            roc_by_detector[det_name] = {}
            for at, data in attacker_rocs.items():
                roc_by_detector[det_name][at] = ROCResult(
                    fpr=np.array(data['fpr']),
                    tpr=np.array(data['tpr']),
                    thresholds=np.array([]),
                    auc=data['auc'],
                    detector_name=det_name,
                    attacker_type=at,
                )
        results['roc_by_detector'] = roc_by_detector

        # Also create comparison views
        attack_types = ['naive_attack', 'systematic_attack', 'adaptive_attack']
        roc_comparison = {}
        for at in attack_types:
            roc_comparison[at] = {}
            for det_name in pass4:
                if at in pass4[det_name]:
                    data = pass4[det_name][at]
                    roc_comparison[at][det_name] = ROCResult(
                        fpr=np.array(data['fpr']),
                        tpr=np.array(data['tpr']),
                        thresholds=np.array([]),
                        auc=data['auc'],
                    )
        results['roc_comparison'] = roc_comparison

    # Figure 3: Efficiency curve
    if pass5 and 'sweep' in pass5:
        results['efficiency_sweep'] = pass5['sweep']

    # Figure 4: Window analysis
    if pass6 and 'window_analysis' in pass6:
        results['window_analysis'] = pass6['window_analysis']

    # Figure 5: Coverage maps
    fingerprints_path = RESULTS_DIR / 'fingerprints.npy'
    if fingerprints_path.exists() and pass3:
        fingerprints = np.load(fingerprints_path)

        # Build coverage maps via PCA projection
        centered = fingerprints - fingerprints.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ Vt[:2].T

            n_bins = 15
            p_min = projected.min(axis=0)
            p_max = projected.max(axis=0)
            p_range = p_max - p_min + 1e-8

            legit_map = np.zeros((n_bins, n_bins))
            attack_map = np.zeros((n_bins, n_bins))

            sessions = generate_all_sessions(
                DOMAIN_PROBES, fingerprints, k=DEFAULT_WINDOW, seed=42,
                n_legit_single=100, n_legit_mixed=100, n_legit_power=50,
                n_attack_naive=100, n_attack_systematic=100, n_attack_adaptive=50,
            )

            for stype, sess_list in sessions.items():
                for session in sess_list:
                    fps = projected[session.probe_indices]
                    for p in fps:
                        bx = min(int((p[0] - p_min[0]) / p_range[0] * n_bins), n_bins - 1)
                        by = min(int((p[1] - p_min[1]) / p_range[1] * n_bins), n_bins - 1)
                        if session.label == 0:
                            legit_map[by, bx] += 1
                        else:
                            attack_map[by, bx] += 1

            results['coverage_maps'] = {
                'legitimate': legit_map,
                'attack': attack_map,
            }
        except np.linalg.LinAlgError:
            print('  WARNING: SVD failed for coverage maps')

    # Figure 6: PoH scores
    if pass7 and 'poh_scores' in pass7:
        results['poh_scores'] = pass7['poh_scores']

    # Figure 7: Signal decomposition
    if pass7 and 'signal_decomposition' in pass7:
        results['signal_decomposition'] = pass7['signal_decomposition']

    generate_all_figures(results, str(RESULTS_DIR))
    print(f'\n  All figures saved to {RESULTS_DIR}/')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Paper 5: Capability Manifold Surveillance (Qwen 1.5B)')
    parser.add_argument('--pass', type=int, dest='run_pass',
                        help='Run specific pass (1-7)')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all passes sequentially')
    parser.add_argument('--figures-only', action='store_true',
                        help='Generate figures from cached results')
    parser.add_argument('--device', default='cpu',
                        help='Device for inference (cpu/cuda/mps)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        generate_figures(args)
        return

    fingerprints = None
    input_embeddings = None
    all_sessions = None
    all_scores = None
    all_detections = None

    if args.run_all or args.run_pass == 1:
        fingerprints, input_embeddings, _ = run_pass1(args)

    if args.run_all or args.run_pass == 2:
        _, _ = run_pass2(args, fingerprints, input_embeddings)

    if args.run_all or args.run_pass == 3:
        all_sessions, all_scores, all_detections, _ = run_pass3(
            args, fingerprints, input_embeddings)

    if args.run_all or args.run_pass == 4:
        run_pass4(args, all_scores)

    if args.run_all or args.run_pass == 5:
        run_pass5(args, fingerprints)

    if args.run_all or args.run_pass == 6:
        run_pass6(args, fingerprints, input_embeddings)

    if args.run_all or args.run_pass == 7:
        run_pass7(args, all_detections, fingerprints)

    if args.run_all:
        generate_figures(args)

    print('\n' + '='*60)
    print('COMPLETE')
    print('='*60)


if __name__ == '__main__':
    main()
