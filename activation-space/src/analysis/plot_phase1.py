"""
Publication-quality plotting for Phase 1: Ensemble Collapse.

Generates plots showing:
1. CKA representational alignment over training steps with regime bands
2. Pairwise CKA heatmap across seed pairs and checkpoints
3. CKA by regime type (box plot with statistical test)
4. Collapse fidelity comparison (added after collapse experiment)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Match Phase 0 color scheme
REGIME_COLORS = {
    'stable': '#2ecc71',
    'chaotic': '#e74c3c',
    'transition': '#f39c12',
}
REGIME_ALPHAS = {'stable': 0.15, 'chaotic': 0.20, 'transition': 0.15}


def _style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _add_regime_bands(ax, steps, regimes):
    """Add colored vertical bands from regime labels."""
    for i, regime in enumerate(regimes):
        x_start = steps[i] if i > 0 else 0
        x_end = steps[i + 1] if i + 1 < len(steps) else steps[i] + 50
        ax.axvspan(x_start, x_end, color=REGIME_COLORS[regime],
                   alpha=REGIME_ALPHAS[regime])


def _regime_legend():
    return [
        mpatches.Patch(color=REGIME_COLORS['stable'], alpha=0.4, label='Stable'),
        mpatches.Patch(color=REGIME_COLORS['chaotic'], alpha=0.4, label='Chaotic'),
        mpatches.Patch(color=REGIME_COLORS['transition'], alpha=0.4, label='Transition'),
    ]


def plot_convergence(data, save_dir=None):
    """Main Phase 1 convergence figure: CKA trajectory + heatmap + box plot."""
    steps = data['checkpoint_steps']
    cka_mean = np.array(data['cka_mean'])
    cka_std = np.array(data['cka_std'])
    regimes = data['regime_labels']
    regime_steps = [0] + data['regime_steps']

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1])

    # --- Panel 1: CKA trajectory with regime bands (full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    _add_regime_bands(ax1, regime_steps, regimes)
    ax1.plot(steps, cka_mean, color='#2c3e50', linewidth=2, label='Mean CKA')
    ax1.fill_between(steps, cka_mean - cka_std, cka_mean + cka_std,
                     alpha=0.2, color='#2c3e50')
    ax1.set_ylim(0.4, 1.05)
    _style_ax(ax1, 'Training Step', 'Linear CKA',
              'Cross-Seed Representational Alignment (CKA)')
    ax1.legend(handles=[ax1.get_lines()[0]] + _regime_legend(),
               loc='lower right', fontsize=9, framealpha=0.9)
    # Annotate key points
    ax1.annotate('Shared init', xy=(0, cka_mean[0]),
                 xytext=(200, cka_mean[0] + 0.03),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    min_idx = np.argmin(cka_mean)
    ax1.annotate(f'Min: {cka_mean[min_idx]:.3f}', xy=(steps[min_idx], cka_mean[min_idx]),
                 xytext=(steps[min_idx] + 100, cka_mean[min_idx] - 0.05),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))

    # --- Panel 2: Pairwise CKA heatmap ---
    ax2 = fig.add_subplot(gs[1, :])
    seed_pairs = data['seed_pairs']
    pair_labels = [f'{p[0]}v{p[1]}' for p in seed_pairs]
    cka_matrix = np.array([data['cka_per_pair'][f'{p[0]}_vs_{p[1]}']
                           for p in seed_pairs])
    im = ax2.imshow(cka_matrix, aspect='auto', cmap='RdYlGn',
                    vmin=0.5, vmax=1.0, interpolation='nearest')
    ax2.set_yticks(range(len(pair_labels)))
    ax2.set_yticklabels(pair_labels, fontsize=9)
    # Show every 5th step on x-axis
    tick_idx = list(range(0, len(steps), 5))
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([steps[i] for i in tick_idx], fontsize=9)
    ax2.set_xlabel('Training Step', fontsize=11)
    ax2.set_title('Pairwise CKA Across Seed Pairs', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='CKA', shrink=0.8)

    # --- Panel 3a: CKA by regime type (box plot) ---
    ax3 = fig.add_subplot(gs[2, 0])
    cka_by_regime = data['cka_by_regime']
    bp_data = [cka_by_regime['chaotic'], cka_by_regime['transition'],
               cka_by_regime['stable']]
    bp_labels = ['Chaotic', 'Transition', 'Stable']
    bp_colors = [REGIME_COLORS['chaotic'], REGIME_COLORS['transition'],
                 REGIME_COLORS['stable']]

    bp = ax3.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], bp_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in bp['medians']:
        median.set_color('#2c3e50')
        median.set_linewidth(2)

    _style_ax(ax3, 'Regime', 'CKA', 'CKA by Regime Type')

    # Add p-value annotation
    stats = data['stats']
    p_mw = stats['cka_mann_whitney_p']
    sig = '***' if p_mw < 0.001 else '**' if p_mw < 0.01 else '*' if p_mw < 0.05 else 'ns'
    y_max = max(max(v) for v in bp_data) + 0.02
    ax3.plot([1, 1, 3, 3], [y_max, y_max + 0.01, y_max + 0.01, y_max],
             color='black', linewidth=1)
    ax3.text(2, y_max + 0.015, f'p={p_mw:.3f} {sig}',
             ha='center', fontsize=10)

    # --- Panel 3b: Summary text ---
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    summary = (
        f"Cross-Seed Representational Analysis\n"
        f"{'='*40}\n\n"
        f"Seeds: {data['seeds']}\n"
        f"Checkpoints: {len(steps)}\n"
        f"Seed pairs: {len(seed_pairs)}\n\n"
        f"CKA trajectory:\n"
        f"  Init (shared):     {cka_mean[0]:.3f}\n"
        f"  Min (chaotic):     {cka_mean.min():.3f}\n"
        f"  Late stable:       {np.mean(cka_by_regime['stable'][-6:]):.3f}\n\n"
        f"Statistical tests:\n"
        f"  Stable vs Chaotic (MW): p={p_mw:.4f}\n"
        f"  Late stable vs early chaotic:\n"
        f"    p={stats['late_stable_vs_early_chaotic_p']:.4f}\n\n"
        f"Interpretation:\n"
        f"  Models diverge during chaotic regimes\n"
        f"  then re-converge in stable regimes,\n"
        f"  supporting ensemble collapse."
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    fig.suptitle('Phase 1: Ensemble Convergence Evidence',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase1_convergence.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase1_convergence.pdf', bbox_inches='tight')
        print(f"Convergence plots saved to {save_path}")

    return fig


def plot_collapse(collapse_data, save_dir=None):
    """Plot collapse experiment results: fidelity comparison + divergence."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    seeds = collapse_data['seeds']
    baseline_losses = collapse_data['baseline_losses']
    collapsed_losses = collapse_data['collapsed_losses']

    # --- Panel 1: Fidelity bar chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(seeds))
    width = 0.35
    bars1 = ax1.bar(x - width/2, baseline_losses, width, label='Baseline (independent)',
                    color='#3498db', alpha=0.7)
    bars2 = ax1.bar(x + width/2, collapsed_losses, width, label='Collapsed',
                    color='#e74c3c', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=10)
    _style_ax(ax1, '', 'Final Loss (step 2000)', 'Fidelity: Baseline vs Collapsed')
    ax1.legend(fontsize=9, framealpha=0.9)

    # --- Panel 2: Difference ---
    ax2 = fig.add_subplot(gs[0, 1])
    diffs = [c - b for b, c in zip(baseline_losses, collapsed_losses)]
    colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in diffs]
    ax2.bar(x, diffs, color=colors, alpha=0.7)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Seed {s}' for s in seeds], fontsize=10)
    _style_ax(ax2, '', 'Loss Difference (collapsed - baseline)',
              'Fidelity Delta')
    # Add p-value
    if 'p_value' in collapse_data:
        ax2.text(0.05, 0.95, f"p = {collapse_data['p_value']:.3f}",
                 transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Panel 3: Post-collapse loss curves ---
    ax3 = fig.add_subplot(gs[1, 0])
    if 'post_collapse_curves' in collapse_data:
        curves = collapse_data['post_collapse_curves']
        for seed, curve in curves.items():
            steps = curve['steps']
            losses = curve['losses']
            ax3.plot(steps, losses, linewidth=1.5, label=f'Seed {seed}', alpha=0.8)
        _style_ax(ax3, 'Training Step', 'Loss',
                  'Post-Collapse Training Curves')
        ax3.legend(fontsize=9, framealpha=0.9)

    # --- Panel 4: Compute savings ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    n_seeds = len(seeds) + 1  # +1 for donor
    total_baseline = n_seeds * 2000
    total_collapsed = 1 * 2000 + len(seeds) * (2000 - collapse_data.get('collapse_step', 1450))
    savings = 1 - total_collapsed / total_baseline
    summary = (
        f"Compute Savings\n"
        f"{'='*35}\n\n"
        f"Baseline: {n_seeds} seeds × 2000 steps\n"
        f"  = {total_baseline:,} total steps\n\n"
        f"Collapse: 1 donor × 2000 steps\n"
        f"  + {len(seeds)} seeds × {2000 - collapse_data.get('collapse_step', 1450)} steps\n"
        f"  = {total_collapsed:,} total steps\n\n"
        f"Savings: {savings:.0%}\n\n"
        f"Fidelity: p = {collapse_data.get('p_value', 'N/A')}\n"
        f"Mean |delta|: {np.mean(np.abs(diffs)):.4f}"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    fig.suptitle('Phase 1: Ensemble Collapse Results',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase1_collapse.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase1_collapse.pdf', bbox_inches='tight')
        print(f"Collapse plots saved to {save_path}")

    return fig


if __name__ == '__main__':
    import sys, os
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'phase1'

    # Plot convergence (from Workstream A)
    rsa_path = results_dir / 'cross_seed_rsa.json'
    if rsa_path.exists():
        with open(rsa_path) as f:
            rsa_data = json.load(f)
        plt.switch_backend('Agg')
        plot_convergence(rsa_data, save_dir=str(results_dir))
        print('Convergence plots done.')

    # Plot collapse results (from Workstream B)
    collapse_path = results_dir / 'collapse_results.json'
    if collapse_path.exists():
        with open(collapse_path) as f:
            collapse_data = json.load(f)
        plot_collapse(collapse_data, save_dir=str(results_dir))
        print('Collapse plots done.')

    if not rsa_path.exists() and not collapse_path.exists():
        print('No results found. Run analysis scripts first.')
