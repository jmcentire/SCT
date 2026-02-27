"""Evaluation metrics for CMS experiments.

ROC curves, AUC, operating points, and comparative analysis
across detection methods.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ROCResult:
    """ROC curve result."""
    fpr: np.ndarray          # False positive rates
    tpr: np.ndarray          # True positive rates
    thresholds: np.ndarray   # Score thresholds
    auc: float               # Area under curve
    detector_name: str = ''
    attacker_type: str = ''


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> ROCResult:
    """Compute ROC curve from scores and binary labels.

    Args:
        scores: (n,) detection scores (higher = more suspicious)
        labels: (n,) binary labels (0 = legitimate, 1 = attack)

    Returns:
        ROCResult with FPR, TPR, thresholds, and AUC
    """
    # Sort by decreasing score
    desc_idx = np.argsort(-scores)
    sorted_scores = scores[desc_idx]
    sorted_labels = labels[desc_idx]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return ROCResult(
            fpr=np.array([0.0, 1.0]),
            tpr=np.array([0.0, 1.0]),
            thresholds=np.array([1.0, 0.0]),
            auc=0.5,
        )

    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)

    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])
    thresholds = np.concatenate([[sorted_scores[0] + 1], sorted_scores])

    # Remove duplicates (keep last occurrence of each FPR value)
    unique_mask = np.concatenate([np.diff(fpr) > 0, [True]])
    fpr = fpr[unique_mask]
    tpr = tpr[unique_mask]
    thresholds = thresholds[unique_mask]

    # AUC via trapezoidal rule
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    auc_val = float(_trapz(tpr, fpr))

    return ROCResult(fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc_val)


def operating_point(roc: ROCResult, target_fpr: float = 0.01) -> dict:
    """Find operating point at target FPR.

    Returns the TPR (detection rate) achievable at the given FPR.
    """
    # Find the highest TPR where FPR <= target
    valid = roc.fpr <= target_fpr
    if not valid.any():
        return {
            'fpr': 0.0,
            'tpr': 0.0,
            'threshold': float(roc.thresholds[0]),
        }

    best_idx = np.where(valid)[0][-1]  # Last valid index (highest TPR)
    return {
        'fpr': float(roc.fpr[best_idx]),
        'tpr': float(roc.tpr[best_idx]),
        'threshold': float(roc.thresholds[best_idx]),
    }


def compare_detectors(
    detector_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    target_fprs: list[float] = None,
) -> dict:
    """Compare multiple detectors on the same evaluation set.

    Args:
        detector_scores: dict mapping detector_name -> scores array
        labels: binary labels
        target_fprs: FPR values to report operating points at

    Returns:
        dict with ROC curves, AUCs, and operating points per detector
    """
    if target_fprs is None:
        target_fprs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]

    results = {}
    for name, scores in detector_scores.items():
        roc = roc_curve(scores, labels)
        roc.detector_name = name

        ops = {}
        for fpr in target_fprs:
            ops[f'fpr_{fpr}'] = operating_point(roc, fpr)

        results[name] = {
            'roc': roc,
            'auc': roc.auc,
            'operating_points': ops,
        }

    return results


def per_attacker_roc(
    legit_scores: np.ndarray,
    attack_scores_by_type: dict[str, np.ndarray],
    detector_name: str = '',
) -> dict[str, ROCResult]:
    """Compute separate ROC curves for each attacker type vs. legitimate.

    Args:
        legit_scores: scores for legitimate sessions
        attack_scores_by_type: dict mapping attack_type -> scores

    Returns:
        dict mapping attack_type -> ROCResult
    """
    results = {}
    for attack_type, attack_scores in attack_scores_by_type.items():
        scores = np.concatenate([legit_scores, attack_scores])
        labels = np.concatenate([
            np.zeros(len(legit_scores)),
            np.ones(len(attack_scores)),
        ])
        roc = roc_curve(scores, labels)
        roc.detector_name = detector_name
        roc.attacker_type = attack_type
        results[attack_type] = roc

    return results


def window_size_analysis(
    evaluate_fn,
    window_sizes: list[int] = None,
) -> dict:
    """Analyze detection performance across window sizes.

    Args:
        evaluate_fn: callable(window_size) -> dict with 'auc', 'tpr_at_1pct_fpr'
        window_sizes: list of window sizes to test

    Returns:
        dict with window_sizes, aucs, tprs
    """
    if window_sizes is None:
        window_sizes = [10, 15, 20, 25, 30, 40, 50, 75, 100]

    aucs = []
    tprs = []
    for ws in window_sizes:
        result = evaluate_fn(ws)
        aucs.append(result['auc'])
        tprs.append(result['tpr_at_1pct_fpr'])

    return {
        'window_sizes': window_sizes,
        'aucs': aucs,
        'tprs_at_1pct_fpr': tprs,
    }


def summary_table(comparison: dict) -> str:
    """Format comparison results as markdown table."""
    lines = []
    lines.append("| Detector | AUC | TPR@1%FPR | TPR@5%FPR |")
    lines.append("|----------|-----|-----------|-----------|")

    for name, data in comparison.items():
        auc = data['auc']
        tpr_1 = data['operating_points'].get('fpr_0.01', {}).get('tpr', 0)
        tpr_5 = data['operating_points'].get('fpr_0.05', {}).get('tpr', 0)
        lines.append(f"| {name:8s} | {auc:.3f} | {tpr_1:.3f} | {tpr_5:.3f} |")

    return '\n'.join(lines)
