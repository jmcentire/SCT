"""Publication-quality figures for Paper 5: Capability Manifold Surveillance.

Figures:
  1. Similarity distribution histograms (legitimate vs attack)
  2. ROC curves for all detectors × attacker types
  3. Coverage-efficiency curve (adversarial bound)
  4. Window size sensitivity analysis
  5. Bloom filter coverage maps
  6. PoH score accumulation over time
  7. Multi-signal decomposition heatmap
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'single_domain': '#2196F3',
    'mixed_domain': '#4CAF50',
    'power_user': '#FF9800',
    'naive_attack': '#F44336',
    'systematic_attack': '#9C27B0',
    'adaptive_attack': '#E91E63',
    'prada': '#607D8B',
    'vardetect': '#795548',
    'cms': '#1565C0',
}


def plot_similarity_distributions(
    session_similarities: dict[str, list[float]],
    save_path: str = None,
    title: str = 'Pairwise Similarity Distributions',
):
    """Figure 1: Histograms of pairwise similarity for each session type.

    Args:
        session_similarities: dict mapping session_type -> list of mean
                              pairwise similarities (one per session)
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    legit_types = ['single_domain', 'mixed_domain', 'power_user']
    attack_types = ['naive_attack', 'systematic_attack', 'adaptive_attack']

    bins = np.linspace(-0.2, 1.0, 50)

    for stype in legit_types:
        if stype in session_similarities:
            vals = session_similarities[stype]
            ax.hist(vals, bins=bins, alpha=0.5, density=True,
                    color=COLORS.get(stype, '#888'),
                    label=stype.replace('_', ' ').title())

    for stype in attack_types:
        if stype in session_similarities:
            vals = session_similarities[stype]
            ax.hist(vals, bins=bins, alpha=0.5, density=True,
                    color=COLORS.get(stype, '#888'),
                    label=stype.replace('_', ' ').title(),
                    linestyle='--')

    ax.set_xlabel('Mean Pairwise Centered Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.axvline(x=0.65, color='black', linestyle=':', alpha=0.5, label='τ_μ = 0.65')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_roc_curves(
    roc_results: dict,
    save_path: str = None,
    title: str = 'Detection ROC Curves',
):
    """Figure 2: ROC curves for each detector and attacker type.

    Args:
        roc_results: nested dict: detector_name -> attacker_type -> ROCResult
    """
    detectors = list(roc_results.keys())
    n_detectors = len(detectors)

    fig, axes = plt.subplots(1, n_detectors, figsize=(5 * n_detectors, 5),
                              squeeze=False)

    for i, detector in enumerate(detectors):
        ax = axes[0, i]
        attacker_rocs = roc_results[detector]

        for attack_type, roc in attacker_rocs.items():
            color = COLORS.get(attack_type, '#888')
            label = f"{attack_type.replace('_', ' ')} (AUC={roc.auc:.3f})"
            ax.plot(roc.fpr, roc.tpr, color=color, linewidth=2, label=label)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{detector.upper()}')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_xlim([0, 0.15])  # Zoom to low-FPR region
        ax.set_ylim([0, 1.05])

    fig.suptitle(title, fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_roc_comparison(
    detector_rocs: dict,
    attack_type: str = 'systematic_attack',
    save_path: str = None,
):
    """ROC comparison across detectors for a single attacker type.

    Args:
        detector_rocs: dict mapping detector_name -> ROCResult
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Full ROC
    for name, roc in detector_rocs.items():
        color = COLORS.get(name, '#888')
        ax1.plot(roc.fpr, roc.tpr, color=color, linewidth=2,
                 label=f'{name.upper()} (AUC={roc.auc:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'Full ROC — {attack_type.replace("_", " ").title()}')
    ax1.legend(loc='lower right')

    # Zoomed to low-FPR region
    for name, roc in detector_rocs.items():
        color = COLORS.get(name, '#888')
        ax2.plot(roc.fpr, roc.tpr, color=color, linewidth=2,
                 label=f'{name.upper()}')

    ax2.axvline(x=0.01, color='red', linestyle=':', alpha=0.5, label='1% FPR')
    ax2.axvline(x=0.05, color='orange', linestyle=':', alpha=0.5, label='5% FPR')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Zoomed: Low-FPR Region')
    ax2.set_xlim([0, 0.10])
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc='lower right')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_efficiency_curve(
    sweep_results: dict,
    save_path: str = None,
):
    """Figure 3: Coverage efficiency vs. detection constraint.

    The key adversarial bound figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tau_values = sweep_results['tau_values']
    coverages = sweep_results['coverages']
    efficiency_ratios = sweep_results['efficiency_ratios']
    baseline = sweep_results['baseline_coverage']

    # Absolute coverage
    ax1.plot(tau_values, coverages, 'o-', color='#1565C0', linewidth=2, markersize=6)
    ax1.axhline(y=baseline, color='red', linestyle='--', alpha=0.5,
                label=f'Unconstrained baseline: {baseline:.3f}')
    ax1.set_xlabel('Detection Threshold τ_μ')
    ax1.set_ylabel('Manifold Coverage')
    ax1.set_title('Coverage vs. Detection Constraint')
    ax1.legend()

    # Relative efficiency
    ax2.plot(tau_values, efficiency_ratios, 's-', color='#E91E63', linewidth=2, markersize=6)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='50% efficiency')
    ax2.set_xlabel('Detection Threshold τ_μ')
    ax2.set_ylabel('Efficiency Ratio (relative to unconstrained)')
    ax2.set_title('Extraction Efficiency Under Constraint')
    ax2.set_ylim([0, 1.1])
    ax2.legend()

    fig.suptitle('Coverage-Clustering Tradeoff', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_window_analysis(
    window_results: dict,
    save_path: str = None,
):
    """Figure 4: Detection performance vs. window size."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ws = window_results['window_sizes']
    aucs = window_results['aucs']
    tprs = window_results['tprs_at_1pct_fpr']

    ax1.plot(ws, aucs, 'o-', color='#1565C0', linewidth=2, markersize=6)
    ax1.set_xlabel('Window Size (k)')
    ax1.set_ylabel('AUC')
    ax1.set_title('Detection AUC vs. Window Size')
    ax1.set_ylim([0.5, 1.05])

    ax2.plot(ws, tprs, 's-', color='#E91E63', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='green', linestyle=':', alpha=0.5, label='95% target')
    ax2.set_xlabel('Window Size (k)')
    ax2.set_ylabel('TPR at 1% FPR')
    ax2.set_title('Detection Sensitivity vs. Window Size')
    ax2.set_ylim([0, 1.05])
    ax2.legend()

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_coverage_maps(
    legit_coverage_map: np.ndarray,
    attack_coverage_map: np.ndarray,
    save_path: str = None,
):
    """Figure 5: Bloom filter / capability coverage maps.

    Side-by-side 2D histograms showing where queries land in capability space.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    vmax = max(legit_coverage_map.max(), attack_coverage_map.max())

    im1 = ax1.imshow(legit_coverage_map, cmap='Blues', aspect='auto',
                     vmin=0, vmax=vmax)
    ax1.set_title('Legitimate Sessions')
    ax1.set_xlabel('Capability Dimension 1')
    ax1.set_ylabel('Capability Dimension 2')
    plt.colorbar(im1, ax=ax1, label='Query density')

    im2 = ax2.imshow(attack_coverage_map, cmap='Reds', aspect='auto',
                     vmin=0, vmax=vmax)
    ax2.set_title('Distillation Attack Sessions')
    ax2.set_xlabel('Capability Dimension 1')
    ax2.set_ylabel('Capability Dimension 2')
    plt.colorbar(im2, ax=ax2, label='Query density')

    fig.suptitle('Capability Space Query Distribution', fontsize=14, y=1.02)

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_poh_scores(
    score_histories: dict[str, list[float]],
    save_path: str = None,
):
    """Figure 6: PoH extraction score accumulation over time.

    Shows how scores evolve for different session types.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for stype, history in score_histories.items():
        color = COLORS.get(stype, '#888')
        label = stype.replace('_', ' ').title()
        ax.plot(history, color=color, linewidth=1.5, alpha=0.8, label=label)

    # Response level thresholds
    for threshold, name in [(0.3, 'Monitor'), (0.5, 'Rate-limit'),
                            (0.7, 'Review'), (0.85, 'Block')]:
        ax.axhline(y=threshold, color='gray', linestyle=':', alpha=0.4)
        ax.text(len(next(iter(score_histories.values()))) * 0.95, threshold + 0.01,
                name, fontsize=8, alpha=0.6, ha='right')

    ax.set_xlabel('Window Number')
    ax.set_ylabel('Extraction Likelihood Score')
    ax.set_title('PoH Score Accumulation by Session Type')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper left')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_signal_decomposition(
    signal_data: dict[str, dict[str, float]],
    save_path: str = None,
):
    """Figure 7: Multi-signal decomposition heatmap.

    Shows which signals fire for each session type.
    """
    session_types = list(signal_data.keys())
    if not session_types:
        return None

    signal_names = list(next(iter(signal_data.values())).keys())
    n_sessions = len(session_types)
    n_signals = len(signal_names)

    matrix = np.zeros((n_sessions, n_signals))
    for i, stype in enumerate(session_types):
        for j, sig in enumerate(signal_names):
            matrix[i, j] = signal_data[stype].get(sig, 0.0)

    fig, ax = plt.subplots(figsize=(8, max(4, n_sessions * 0.5 + 1)))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(n_signals))
    ax.set_xticklabels([s.replace('_', '\n') for s in signal_names], fontsize=9)
    ax.set_yticks(range(n_sessions))
    ax.set_yticklabels([s.replace('_', ' ').title() for s in session_types], fontsize=9)

    # Annotate cells
    for i in range(n_sessions):
        for j in range(n_signals):
            val = matrix[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=8)

    plt.colorbar(im, ax=ax, label='Signal Intensity')
    ax.set_title('Detection Signal Decomposition by Session Type')

    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def generate_all_figures(results: dict, output_dir: str = 'results/paper5'):
    """Generate all Paper 5 figures from experiment results.

    Args:
        results: dict with all experiment outputs
        output_dir: directory for saving figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if 'similarity_distributions' in results:
        plot_similarity_distributions(
            results['similarity_distributions'],
            save_path=f'{output_dir}/fig1_similarity_distributions.png')

    if 'roc_by_detector' in results:
        plot_roc_curves(
            results['roc_by_detector'],
            save_path=f'{output_dir}/fig2_roc_curves.png')

    if 'roc_comparison' in results:
        for attack_type, rocs in results['roc_comparison'].items():
            plot_roc_comparison(
                rocs, attack_type=attack_type,
                save_path=f'{output_dir}/fig2b_roc_comparison_{attack_type}.png')

    if 'efficiency_sweep' in results:
        plot_efficiency_curve(
            results['efficiency_sweep'],
            save_path=f'{output_dir}/fig3_efficiency_curve.png')

    if 'window_analysis' in results:
        plot_window_analysis(
            results['window_analysis'],
            save_path=f'{output_dir}/fig4_window_analysis.png')

    if 'coverage_maps' in results:
        plot_coverage_maps(
            results['coverage_maps']['legitimate'],
            results['coverage_maps']['attack'],
            save_path=f'{output_dir}/fig5_coverage_maps.png')

    if 'poh_scores' in results:
        plot_poh_scores(
            results['poh_scores'],
            save_path=f'{output_dir}/fig6_poh_scores.png')

    if 'signal_decomposition' in results:
        plot_signal_decomposition(
            results['signal_decomposition'],
            save_path=f'{output_dir}/fig7_signal_decomposition.png')
