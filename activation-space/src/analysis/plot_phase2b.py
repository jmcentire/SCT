"""
Publication-quality plotting for Phase 2b: Enhanced Speculative Training.

6-panel figure:
1. Loss + predictions (full width): regime bands + markers for all predictor types
2. Predictor comparison (left): grouped bar chart — linear vs momentum vs quadratic
3. Prediction scatter (right): predicted vs current loss, colored by regime, shaped by predictor
4. Landscape probing (full width): directional derivative + trust radius vs step
5. Worker simulation (left): L2 distance vs step, threshold line, accepted/rejected markers
6. Summary panel (right): all stats including hypothetical parallel speedup
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# Match Phase 0/1/2 color scheme
REGIME_COLORS = {
    'stable': '#2ecc71',
    'chaotic': '#e74c3c',
    'transition': '#f39c12',
    'unknown': '#95a5a6',
}
REGIME_ALPHAS = {'stable': 0.15, 'chaotic': 0.20, 'transition': 0.15, 'unknown': 0.10}

PREDICTOR_COLORS = {
    'linear': '#3498db',
    'momentum': '#9b59b6',
    'quadratic': '#e67e22',
}
PREDICTOR_MARKERS = {
    'linear': 'o',
    'momentum': 's',
    'quadratic': '^',
}


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


# ============================================================
# Panel 1: Loss with Predictions (all predictor types)
# ============================================================

def plot_loss_with_predictions(data, ax=None):
    """Plot training loss with regime bands and prediction markers for all predictors."""
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

    # Applied prediction markers (different colors by predictor)
    applied = data.get('applied_predictions', [])
    applied_steps = set()

    for pred in applied:
        s = pred['step']
        K = pred['K']
        predictor = pred.get('predictor', 'linear')
        applied_steps.add(s)
        color = PREDICTOR_COLORS.get(predictor, '#2ecc71')
        marker = PREDICTOR_MARKERS.get(predictor, 'v')

        if s in steps:
            idx = steps.index(s)
            y = losses[idx]
            ax.scatter(s, y, color=color, marker='v', s=120, zorder=5,
                       edgecolors='white', linewidths=1)
            ax.annotate('', xy=(s + K, y - 0.02), xytext=(s, y),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2, ls='--'))
            ax.text(s + K/2, y - 0.05, f'{predictor[:3]} K={K}', fontsize=7,
                    ha='center', color=color, fontweight='bold')

    # Red X for stable-confirmed rejections
    for ckpt in data.get('checkpoint_log', []):
        s = ckpt['step']
        if s not in applied_steps and ckpt.get('k_results'):
            any_evaluated = any(
                v.get('predicted_loss') is not None
                for v in ckpt['k_results'].values()
            )
            if any_evaluated and s in steps and ckpt.get('is_stable_confirmed'):
                idx = steps.index(s)
                y = losses[idx]
                ax.scatter(s, y, color='#e74c3c', marker='x', s=80, zorder=4, linewidths=2)

    _style_ax(ax, 'Training Step', 'Loss',
              'Training Loss with Enhanced Predictions')

    # Legend
    legend_elements = [ax.get_lines()[0]] + _regime_legend()
    for pred_name, color in PREDICTOR_COLORS.items():
        legend_elements.append(plt.scatter([], [], color=color, marker='v', s=60,
                                           label=f'{pred_name.capitalize()} leap'))
    legend_elements.append(plt.scatter([], [], color='#e74c3c', marker='x', s=60,
                                       label='Rejected (stable)'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9, ncol=2)

    return ax


# ============================================================
# Panel 2: Predictor Comparison Bar Chart
# ============================================================

def plot_predictor_comparison(data, ax=None):
    """Grouped bar chart: success rate by predictor type and K, stable regime only."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    comparison = data.get('predictor_comparison', {})
    predictors = ['linear', 'momentum', 'quadratic']
    k_values = sorted([int(k) for k in data.get('config', {}).get('k_values', [5, 10, 20, 50])])

    if not comparison:
        ax.text(0.5, 0.5, 'No predictor comparison data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return ax

    x = np.arange(len(k_values))
    width = 0.25
    offsets = [-width, 0, width]

    for i, pred in enumerate(predictors):
        rates = []
        counts = []
        for K in k_values:
            pred_data = comparison.get(pred, {}).get(str(K), {})
            stable_data = pred_data.get('stable', {})
            rates.append(stable_data.get('rate', 0) * 100)
            counts.append(stable_data.get('total', 0))

        color = PREDICTOR_COLORS.get(pred, '#95a5a6')
        total_n = sum(counts)
        bars = ax.bar(x + offsets[i], rates, width,
                      label=f'{pred.capitalize()} (n={total_n})',
                      color=color, alpha=0.7,
                      edgecolor='white', linewidth=0.5)

        for bar, n in zip(bars, counts):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'n={n}', ha='center', fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={K}' for K in k_values], fontsize=10)
    ax.set_ylim(0, 105)
    _style_ax(ax, 'Prediction Horizon (K steps)', 'Success Rate (%)',
              'Predictor Comparison (Stable Regime)')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


# ============================================================
# Panel 3: Prediction Scatter (shaped by predictor)
# ============================================================

def plot_prediction_scatter(data, ax=None):
    """Scatter: predicted vs current loss, colored by regime, shaped by predictor."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    predictions = data.get('prediction_log', [])
    valid = [p for p in predictions if p.get('predicted_val_loss') is not None]

    if not valid:
        ax.text(0.5, 0.5, 'No prediction data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return ax

    for p in valid:
        color = REGIME_COLORS.get(p['regime'], '#95a5a6')
        marker = PREDICTOR_MARKERS.get(p.get('predictor', 'linear'), 'o')
        ax.scatter(p['current_val_loss'], p['predicted_val_loss'],
                   color=color, marker=marker, s=25, alpha=0.5,
                   edgecolors='white', linewidths=0.3)

    # Diagonal reference
    all_losses = [p['current_val_loss'] for p in valid] + [p['predicted_val_loss'] for p in valid]
    lo, hi = min(all_losses) - 0.1, max(all_losses) + 0.1
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.fill_between([lo, hi], [lo, hi], [lo, lo], color='#2ecc71', alpha=0.05)

    _style_ax(ax, 'Current Validation Loss', 'Predicted Validation Loss',
              'Prediction Quality (below diagonal = success)')

    # Combined legend: regime colors + predictor shapes
    legend_handles = _regime_legend()
    for pred, marker in PREDICTOR_MARKERS.items():
        legend_handles.append(plt.scatter([], [], color='gray', marker=marker, s=40,
                                          label=pred.capitalize()))
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9, loc='upper left')
    ax.set_aspect('equal')

    return ax


# ============================================================
# Panel 4: Landscape Probing
# ============================================================

def plot_landscape_probing(data, ax=None):
    """Directional derivative + trust radius vs step with regime bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    probe_log = data.get('landscape_probe_log', [])
    regime_steps = data['regime_timeline']['steps']
    regimes = data['regime_timeline']['regimes']

    _add_regime_bands(ax, regime_steps, regimes)

    if not probe_log:
        ax.text(0.5, 0.5, 'No landscape probe data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, 'Training Step', '', 'Landscape Probing')
        return ax

    steps = [p['step'] for p in probe_log]
    derivatives = [p['directional_derivative'] for p in probe_log]
    trust_radii = [p['trust_radius'] for p in probe_log]

    # Left y-axis: directional derivative
    color_deriv = '#e74c3c'
    ax.plot(steps, derivatives, color=color_deriv, linewidth=1.5, marker='o', markersize=4,
            alpha=0.8, label='Directional Derivative')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Directional Derivative', color=color_deriv, fontsize=11)
    ax.tick_params(axis='y', labelcolor=color_deriv)

    # Right y-axis: trust radius
    ax2 = ax.twinx()
    color_trust = '#3498db'
    ax2.plot(steps, trust_radii, color=color_trust, linewidth=1.5, marker='s', markersize=4,
             alpha=0.8, label='Trust Radius (max K)')
    ax2.set_ylabel('Trust Radius (max K)', color=color_trust, fontsize=11)
    ax2.tick_params(axis='y', labelcolor=color_trust)
    ax2.spines['top'].set_visible(False)

    ax.set_xlabel('Training Step', fontsize=11)
    ax.set_title('Landscape Probing: Directional Derivative + Trust Radius', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9, framealpha=0.9)

    return ax


# ============================================================
# Panel 5: Worker Simulation
# ============================================================

def plot_worker_simulation(data, ax=None):
    """L2 distance to actual vs step, threshold line, accepted/rejected markers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    worker_log = data.get('worker_results_log', [])
    threshold = data.get('config', {}).get('worker_handoff_threshold', 0.05)

    if not worker_log:
        ax.text(0.5, 0.5, 'No worker simulation data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, 'Training Step', 'Relative L2 Distance', 'Worker Simulation')
        return ax

    # Plot each worker as a point
    for w in worker_log:
        color = PREDICTOR_COLORS.get(w['predictor_type'], '#95a5a6')
        marker = 'v' if w['handoff_accepted'] else 'x'
        edge_color = '#2ecc71' if w['handoff_accepted'] else '#e74c3c'
        ax.scatter(w['step'], w['relative_l2'], color=color, marker=marker, s=80,
                   edgecolors=edge_color, linewidths=2, zorder=4, alpha=0.8)

    # Threshold line
    steps = [w['step'] for w in worker_log]
    if steps:
        ax.axhline(y=threshold, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7,
                   label=f'Handoff threshold ({threshold})')

    _style_ax(ax, 'Training Step', 'Relative L2 Distance',
              'Worker Simulation: Handoff Distance')

    # Legend: predictor types + accepted/rejected
    legend_handles = []
    for pred, color in PREDICTOR_COLORS.items():
        legend_handles.append(plt.scatter([], [], color=color, s=50, label=pred.capitalize()))
    legend_handles.append(plt.scatter([], [], color='gray', marker='v', s=50,
                                      edgecolors='#2ecc71', linewidths=2, label='Accepted'))
    legend_handles.append(plt.scatter([], [], color='gray', marker='x', s=50,
                                      edgecolors='#e74c3c', linewidths=2, label='Rejected'))
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9)

    return ax


# ============================================================
# Panel 6: Summary
# ============================================================

def plot_summary(data, ax=None):
    """Summary statistics panel with all Phase 2b metrics."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.axis('off')

    s = data.get('summary', {})
    cfg = data.get('config', {})

    trained = s.get('total_steps_trained', 0)
    skipped = s.get('total_steps_skipped', 0)
    effective = s.get('effective_total', 0)
    speedup = s.get('effective_speedup', 0)
    hyp_speedup = s.get('hypothetical_parallel_speedup', 0)
    final_val = s.get('final_val_loss', float('inf'))
    final_train = s.get('final_train_loss', float('inf'))
    baseline = cfg.get('phase0_baseline_loss', 1.756)
    n_applied = s.get('predictions_applied', 0)
    total_time = s.get('total_time_seconds', 0)
    best_pred = s.get('best_predictor', 'N/A')
    best_rate = s.get('best_predictor_stable_rate', 0)
    n_workers = s.get('workers_evaluated', 0)
    n_accepted = s.get('workers_accepted', 0)
    cache_hits = s.get('cache_hits', 0)
    cache_inserts = s.get('cache_inserts', 0)

    # Regime distribution
    regimes = data.get('regime_timeline', {}).get('regimes', [])
    n_stable = regimes.count('stable')
    n_chaotic = regimes.count('chaotic')
    n_transition = regimes.count('transition')
    n_total = len(regimes)

    summary = (
        f"Phase 2b: Enhanced Speculative Summary\n"
        f"{'=' * 42}\n\n"
        f"Steps trained (gradient): {trained:>6}\n"
        f"Steps skipped (predicted): {skipped:>5}\n"
        f"Effective total:          {effective:>6}\n"
        f"Effective speedup:       {speedup:>6.1%}\n"
        f"Hypothetical parallel:   {hyp_speedup:>6.1%}\n\n"
        f"Best predictor:   {best_pred:>14}\n"
        f"  Stable rate:    {best_rate:>13.1%}\n"
        f"Predictions applied:     {n_applied:>6}\n"
        f"Total time:              {total_time:>5.0f}s\n\n"
        f"Workers eval'd / accept: {n_workers:>3} / {n_accepted}\n"
        f"Cache inserts / hits:    {cache_inserts:>3} / {cache_hits}\n\n"
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
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    return ax


# ============================================================
# Master Plot Function
# ============================================================

def plot_all(data, save_dir=None, show=False):
    """Generate all Phase 2b plots in a 6-panel figure."""
    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1, 1])

    # Row 1: Loss with predictions (full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_loss_with_predictions(data, ax=ax1)

    # Row 2: Predictor comparison (left), Prediction scatter (right)
    ax2 = fig.add_subplot(gs[1, 0])
    plot_predictor_comparison(data, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_prediction_scatter(data, ax=ax3)

    # Row 3: Landscape probing (full width)
    ax4 = fig.add_subplot(gs[2, :])
    plot_landscape_probing(data, ax=ax4)

    # Row 4: Worker simulation (left), Summary (right)
    ax5 = fig.add_subplot(gs[3, 0])
    plot_worker_simulation(data, ax=ax5)

    ax6 = fig.add_subplot(gs[3, 1])
    plot_summary(data, ax=ax6)

    fig.suptitle('Phase 2b: Enhanced Speculative Training',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase2b_enhanced.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase2b_enhanced.pdf', bbox_inches='tight')
        print(f"Phase 2b plots saved to {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    results_dir = Path(__file__).parent.parent.parent / 'results' / 'phase2b'
    results_path = results_dir / 'enhanced_results.json'

    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        plot_all(data, save_dir=str(results_dir))
        print('Phase 2b plots done.')
    else:
        print(f'No results found at {results_path}. Run phase2b_enhanced_speculative.py first.')
