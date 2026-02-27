"""
Publication-quality plotting for Phase 2: Speculative Training (Leap+Verify).

Generates plots showing:
1. Training loss with regime bands + prediction markers
2. Success rate vs K, grouped by regime type (bar chart)
3. Predicted loss vs current loss scatter (colored by regime)
4. Summary panel: steps trained/skipped, effective speedup, final loss, fidelity
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Match Phase 0/1 color scheme
REGIME_COLORS = {
    'stable': '#2ecc71',
    'chaotic': '#e74c3c',
    'transition': '#f39c12',
    'unknown': '#95a5a6',
}
REGIME_ALPHAS = {'stable': 0.15, 'chaotic': 0.20, 'transition': 0.15, 'unknown': 0.10}


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
        x_start = steps[i - 1] if i > 0 else 0
        x_end = steps[i]
        color = REGIME_COLORS.get(regime, '#95a5a6')
        alpha = REGIME_ALPHAS.get(regime, 0.10)
        ax.axvspan(x_start, x_end, color=color, alpha=alpha)


def _regime_legend():
    return [
        mpatches.Patch(color=REGIME_COLORS['stable'], alpha=0.4, label='Stable'),
        mpatches.Patch(color=REGIME_COLORS['chaotic'], alpha=0.4, label='Chaotic'),
        mpatches.Patch(color=REGIME_COLORS['transition'], alpha=0.4, label='Transition'),
    ]


def plot_loss_with_predictions(data, ax=None):
    """Plot training loss with regime bands and prediction markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    # Regime bands
    regime_steps = data['regime_timeline']['steps']
    regimes = data['regime_timeline']['regimes']
    _add_regime_bands(ax, regime_steps, regimes)

    # Loss curve
    steps = data['loss_curve']['steps']
    losses = data['loss_curve']['values']
    ax.plot(steps, losses, color='#2c3e50', linewidth=1.5, label='Training Loss', zorder=2)

    # Prediction markers
    applied = data.get('applied_predictions', [])
    rejected_steps = set()
    applied_steps = set()

    # Mark applied predictions (green arrows)
    for pred in applied:
        s = pred['step']
        K = pred['K']
        applied_steps.add(s)
        # Find the loss at this step
        if s in steps:
            idx = steps.index(s)
            y = losses[idx]
            # Green check marker
            ax.scatter(s, y, color='#2ecc71', marker='v', s=120, zorder=5,
                       edgecolors='white', linewidths=1)
            # Arrow showing skip
            ax.annotate('', xy=(s + K, y - 0.02), xytext=(s, y),
                        arrowprops=dict(arrowstyle='->', color='#2ecc71',
                                        lw=2, ls='--'))
            ax.text(s + K/2, y - 0.05, f'K={K}', fontsize=8, ha='center',
                    color='#2ecc71', fontweight='bold')

    # Mark rejected predictions at checkpoints where we had predictions but didn't apply
    for ckpt in data.get('checkpoint_log', []):
        s = ckpt['step']
        if s not in applied_steps and ckpt.get('k_results'):
            # Check if any predictions were evaluated
            any_evaluated = any(
                v.get('predicted_loss') is not None
                for v in ckpt['k_results'].values()
            )
            if any_evaluated and s in steps:
                idx = steps.index(s)
                y = losses[idx]
                # Only show red X if we were in stable-confirmed but nothing worked,
                # or just show a small dot for non-stable checkpoints
                if ckpt.get('is_stable_confirmed'):
                    ax.scatter(s, y, color='#e74c3c', marker='x', s=80, zorder=4,
                               linewidths=2)

    _style_ax(ax, 'Training Step', 'Loss',
              'Training Loss with Regime Detection + Leap Predictions')

    # Legend
    legend_elements = [ax.get_lines()[0]] + _regime_legend()
    legend_elements.append(plt.scatter([], [], color='#2ecc71', marker='v', s=80,
                                       label='Leap accepted'))
    legend_elements.append(plt.scatter([], [], color='#e74c3c', marker='x', s=80,
                                       label='Leap rejected (stable)'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    return ax


def plot_success_by_k_regime(data, ax=None):
    """Bar chart: success rate vs K, grouped by regime."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    success_data = data.get('success_by_k_regime', {})
    k_values = sorted([int(k) for k in success_data.keys()])
    regime_order = ['chaotic', 'transition', 'stable']

    if not k_values:
        ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return ax

    x = np.arange(len(k_values))
    width = 0.25
    offsets = [-width, 0, width]

    for i, regime in enumerate(regime_order):
        rates = []
        counts = []
        for K in k_values:
            r_data = success_data.get(str(K), {}).get(regime, {})
            rates.append(r_data.get('rate', 0) * 100)
            counts.append(r_data.get('total', 0))

        bars = ax.bar(x + offsets[i], rates, width,
                       label=f'{regime.capitalize()} (n={sum(counts)})',
                       color=REGIME_COLORS[regime], alpha=0.7,
                       edgecolor='white', linewidth=0.5)

        # Add count labels on bars
        for bar, n in zip(bars, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'n={n}', ha='center', fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={K}' for K in k_values], fontsize=10)
    ax.set_ylim(0, 105)
    _style_ax(ax, 'Prediction Horizon (K steps)', 'Success Rate (%)',
              'Prediction Success Rate by K and Regime')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


def plot_prediction_scatter(data, ax=None):
    """Scatter: predicted loss vs current loss, colored by regime."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    predictions = data.get('prediction_log', [])
    if not predictions:
        ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return ax

    # Filter to entries with actual predictions
    valid = [p for p in predictions if p.get('predicted_val_loss') is not None]

    for p in valid:
        color = REGIME_COLORS.get(p['regime'], '#95a5a6')
        marker = 'o'
        ax.scatter(p['current_val_loss'], p['predicted_val_loss'],
                   color=color, marker=marker, s=30, alpha=0.6,
                   edgecolors='white', linewidths=0.5)

    # Diagonal reference (below = successful)
    if valid:
        all_losses = [p['current_val_loss'] for p in valid] + [p['predicted_val_loss'] for p in valid]
        lo, hi = min(all_losses) - 0.1, max(all_losses) + 0.1
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # Shade below diagonal (success zone)
        ax.fill_between([lo, hi], [lo, hi], [lo, lo], color='#2ecc71', alpha=0.05)
        ax.text(hi - 0.05, lo + 0.05, 'Success\nzone', fontsize=9,
                color='#2ecc71', ha='right', va='bottom', alpha=0.7)

    _style_ax(ax, 'Current Validation Loss', 'Predicted Validation Loss',
              'Prediction Quality (below diagonal = success)')
    ax.legend(handles=_regime_legend(), fontsize=9, framealpha=0.9, loc='upper left')
    ax.set_aspect('equal')

    return ax


def plot_summary(data, ax=None):
    """Summary statistics panel."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.axis('off')

    s = data.get('summary', {})
    cfg = data.get('config', {})

    trained = s.get('total_steps_trained', 0)
    skipped = s.get('total_steps_skipped', 0)
    effective = s.get('effective_total', 0)
    speedup = s.get('effective_speedup', 0)
    final_val = s.get('final_val_loss', float('inf'))
    final_train = s.get('final_train_loss', float('inf'))
    baseline = cfg.get('phase0_baseline_loss', 1.756)
    n_applied = s.get('predictions_applied', 0)
    total_time = s.get('total_time_seconds', 0)

    # Count regime distribution
    regimes = data.get('regime_timeline', {}).get('regimes', [])
    n_stable = regimes.count('stable')
    n_chaotic = regimes.count('chaotic')
    n_transition = regimes.count('transition')
    n_total = len(regimes)

    summary = (
        f"Phase 2: Leap+Verify Summary\n"
        f"{'=' * 38}\n\n"
        f"Steps trained (gradient): {trained:>6}\n"
        f"Steps skipped (predicted): {skipped:>5}\n"
        f"Effective total:          {effective:>6}\n"
        f"Effective speedup:       {speedup:>6.1%}\n\n"
        f"Predictions applied:     {n_applied:>6}\n"
        f"Total time:              {total_time:>5.0f}s\n\n"
        f"Final val loss:          {final_val:>7.4f}\n"
        f"Final train loss:        {final_train:>7.4f}\n"
        f"Phase 0 baseline:        {baseline:>7.4f}\n\n"
        f"Regime distribution:\n"
        f"  Stable:     {n_stable:>3} / {n_total}"
        f" ({100*n_stable/max(n_total,1):.0f}%)\n"
        f"  Chaotic:    {n_chaotic:>3} / {n_total}"
        f" ({100*n_chaotic/max(n_total,1):.0f}%)\n"
        f"  Transition: {n_transition:>3} / {n_total}"
        f" ({100*n_transition/max(n_total,1):.0f}%)\n"
    )

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    return ax


def plot_all(data, save_dir=None, show=False):
    """Generate all Phase 2 plots in a single figure."""
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1])

    # Row 1: Loss with predictions (full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_loss_with_predictions(data, ax=ax1)

    # Row 2: Success rate by K (left), Prediction scatter (right)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_success_by_k_regime(data, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_prediction_scatter(data, ax=ax3)

    # Row 3: Summary (full width, centered)
    ax4 = fig.add_subplot(gs[2, :])
    plot_summary(data, ax=ax4)

    fig.suptitle('Phase 2: Speculative Training (Leap+Verify)',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase2_leap_verify.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase2_leap_verify.pdf', bbox_inches='tight')
        print(f"Phase 2 plots saved to {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    results_dir = Path(__file__).parent.parent.parent / 'results' / 'phase2'
    results_path = results_dir / 'leap_verify_results.json'

    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        plot_all(data, save_dir=str(results_dir))
        print('Phase 2 plots done.')
    else:
        print(f'No results found at {results_path}. Run phase2_leap_verify.py first.')
