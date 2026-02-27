"""
Publication-quality plotting for Phase 2c: Aggressive Speculative Depth.

6-panel figure from aggregate.json:
1. Depth Curve (full width): acceptance rate vs K, per predictor, error bands
2. Cascade Chain (left): L2 drift vs chain link number
3. Acceptance Comparison (right): strict vs adaptive vs pct grouped bars
4. Landscape V2 (full width): loss along prediction direction at multiple scales
5. Cross-Seed Consistency (left): per-seed dots + mean for K=5 and K=25
6. Summary (right): headline numbers with error bars
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


REGIME_COLORS = {
    'stable': '#2ecc71',
    'chaotic': '#e74c3c',
    'transition': '#f39c12',
    'unknown': '#95a5a6',
}

PREDICTOR_COLORS = {
    'momentum': '#9b59b6',
    'linear': '#3498db',
    'quadratic': '#e67e22',
}

PREDICTOR_MARKERS = {
    'momentum': 's',
    'linear': 'o',
    'quadratic': '^',
}

CRITERION_COLORS = {
    'strict': '#2c3e50',
    'adaptive': '#27ae60',
    'pct': '#e74c3c',
}

CASCADE_COLORS = {
    '4x25': '#9b59b6',
    '2x50': '#3498db',
    '10x10': '#e67e22',
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


# ============================================================
# Panel 1: Depth Curve
# ============================================================

def plot_depth_curve(data, ax=None):
    """Acceptance rate vs K, one line per predictor, shaded error bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))

    by_pred = data.get('depth_curve', {}).get('by_predictor', {})
    k_values = [5, 10, 25, 50, 75, 100]

    for predictor in ['momentum', 'linear', 'quadratic']:
        pred_data = by_pred.get(predictor, {})
        color = PREDICTOR_COLORS.get(predictor, '#95a5a6')
        marker = PREDICTOR_MARKERS.get(predictor, 'o')

        for regime, ls in [('stable', '-'), ('transition', '--')]:
            regime_data = pred_data.get(regime, {}).get('strict', {})
            means = regime_data.get('mean', [])
            stds = regime_data.get('std', [])

            if not means:
                continue

            ks = k_values[:len(means)]
            means = np.array(means) * 100
            stds = np.array(stds) * 100

            label = f'{predictor.capitalize()} ({regime})'
            ax.plot(ks, means, color=color, marker=marker, markersize=6,
                    linewidth=2, linestyle=ls, label=label, alpha=0.9)
            ax.fill_between(ks, means - stds, means + stds,
                           color=color, alpha=0.15)

    ax.axhline(y=50, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_ylim(-5, 105)
    ax.set_xticks(k_values)
    _style_ax(ax, 'Prediction Horizon K (steps)', 'Acceptance Rate (%)',
              'Speculative Depth Curve: How Far Can We Safely Predict?')
    ax.legend(fontsize=8, framealpha=0.9, ncol=2, loc='upper right')

    return ax


# ============================================================
# Panel 2: Cascade Chain
# ============================================================

def plot_cascade_chain(data, ax=None):
    """L2 drift vs chain link number, one line per cascade config."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    cascade = data.get('cascade_summary', {})

    if not cascade:
        ax.text(0.5, 0.5, 'No cascade data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, 'Chain Link', 'L2 Drift', 'Cascade Chain')
        return ax

    for key, cdata in cascade.items():
        color = CASCADE_COLORS.get(key, '#95a5a6')
        means = cdata.get('mean_l2_per_link', [])
        stds = cdata.get('std_l2_per_link', [])

        if not means:
            continue

        links = np.arange(1, len(means) + 1)
        means = np.array(means)
        stds = np.array(stds)

        ax.errorbar(links, means, yerr=stds, color=color, marker='o',
                    markersize=6, linewidth=2, capsize=4, label=key)

    _style_ax(ax, 'Chain Link Number', 'Relative L2 Drift',
              'Cascade Pipeline: Drift Accumulation')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


# ============================================================
# Panel 3: Acceptance Comparison
# ============================================================

def plot_acceptance_comparison(data, ax=None):
    """Grouped bars: strict vs adaptive vs pct at each K."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    comparison = data.get('acceptance_comparison', {})
    k_values = [5, 10, 25, 50, 75, 100]

    if not comparison:
        ax.text(0.5, 0.5, 'No acceptance data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, '', '', 'Acceptance Criteria Comparison')
        return ax

    criteria = ['strict', 'adaptive', 'pct']
    labels = ['Strict (<)', 'Adaptive (+stddev)', 'Percentage (0.5%)']
    x = np.arange(len(k_values))
    width = 0.25
    offsets = [-width, 0, width]

    for i, (criterion, label) in enumerate(zip(criteria, labels)):
        cdata = comparison.get(criterion, {})
        means = np.array(cdata.get('mean', [0]*len(k_values))) * 100
        stds = np.array(cdata.get('std', [0]*len(k_values))) * 100

        color = CRITERION_COLORS.get(criterion, '#95a5a6')
        ax.bar(x + offsets[i], means[:len(k_values)], width, yerr=stds[:len(k_values)],
               color=color, alpha=0.7, label=label, capsize=3,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'K={K}' for K in k_values], fontsize=9)
    ax.set_ylim(0, 110)
    _style_ax(ax, 'Prediction Horizon K', 'Acceptance Rate (%)',
              'Acceptance Criteria Comparison (Momentum, Stable)')
    ax.legend(fontsize=8, framealpha=0.9)

    return ax


# ============================================================
# Panel 4: Landscape V2
# ============================================================

def plot_landscape_v2(data, ax=None):
    """Loss along prediction direction at multiple scale fractions, per regime."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    # For aggregate: use per-seed combined results if available
    # For single-seed: use landscape_probe_v2 directly
    probes = data.get('landscape_probe_v2', [])

    if not probes:
        ax.text(0.5, 0.5, 'No landscape V2 data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, '', '', 'Landscape Probing V2')
        return ax

    fractions = probes[0].get('fractions', [0.25, 0.5, 1.0, 2.0])

    for regime in ['stable', 'transition']:
        regime_probes = [p for p in probes if p.get('regime') == regime]
        if not regime_probes:
            continue

        color = REGIME_COLORS.get(regime, '#95a5a6')

        # Average losses at each fraction across all probes in this regime
        all_losses = []
        for p in regime_probes:
            losses = p.get('losses_at_fractions', [])
            origin = p.get('loss_at_origin', 0)
            if losses and origin:
                # Normalize: show relative to origin
                normalized = [(l - origin) for l in losses]
                all_losses.append(normalized)

        if all_losses:
            all_losses = np.array(all_losses)
            means = all_losses.mean(axis=0)
            stds = all_losses.std(axis=0)

            ax.plot(fractions, means, color=color, marker='o', markersize=6,
                    linewidth=2, label=f'{regime.capitalize()} (n={len(regime_probes)})')
            ax.fill_between(fractions, means - stds, means + stds,
                           color=color, alpha=0.15)

    ax.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    _style_ax(ax, 'Fraction of Prediction Direction', 'Loss Change from Origin',
              'Landscape V2: Loss Along Prediction Direction')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


# ============================================================
# Panel 5: Cross-Seed Consistency
# ============================================================

def plot_cross_seed(data, ax=None):
    """Per-seed acceptance rates as dots + mean for K=5 and K=25."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    by_pred = data.get('depth_curve', {}).get('by_predictor', {})
    mom_stable = by_pred.get('momentum', {}).get('stable', {}).get('strict', {})
    per_seed = mom_stable.get('per_seed', {})

    if not per_seed:
        ax.text(0.5, 0.5, 'No cross-seed data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        _style_ax(ax, '', '', 'Cross-Seed Consistency')
        return ax

    k_targets = ['5', '25', '50']
    x_positions = np.arange(len(k_targets))

    for i, K_str in enumerate(k_targets):
        seed_rates = per_seed.get(K_str, [])
        if not seed_rates:
            continue

        rates = np.array(seed_rates) * 100

        # Individual seed dots (jittered)
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(rates))
        ax.scatter(x_positions[i] + jitter, rates, color='#3498db', s=50,
                   alpha=0.6, zorder=3, edgecolors='white', linewidths=0.5)

        # Mean + std bar
        mean = np.mean(rates)
        std = np.std(rates)
        ax.errorbar(x_positions[i], mean, yerr=std, color='#2c3e50',
                    marker='D', markersize=8, capsize=6, linewidth=2, zorder=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'K={K}' for K in k_targets], fontsize=11)
    ax.set_ylim(-5, 105)
    _style_ax(ax, 'Prediction Horizon', 'Acceptance Rate (%)',
              'Cross-Seed Consistency (Momentum, Stable)')

    # Custom legend
    ax.scatter([], [], color='#3498db', s=50, alpha=0.6, label='Individual seeds')
    ax.errorbar([], [], yerr=[], color='#2c3e50', marker='D', markersize=8,
                capsize=6, linewidth=2, label='Mean +/- std')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


# ============================================================
# Panel 6: Summary
# ============================================================

def plot_summary_panel(data, ax=None):
    """Summary panel with headline numbers."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.axis('off')

    s = data.get('summary', {})
    n_seeds = s.get('n_seeds', 0)
    max_safe_K = s.get('max_safe_K', 0)
    best_pred = s.get('best_predictor', 'N/A')
    mean_speedup = s.get('mean_hypothetical_speedup', 0)
    std_speedup = s.get('std_hypothetical_speedup', 0)

    # Depth curve highlights
    by_pred = data.get('depth_curve', {}).get('by_predictor', {})
    mom_stable = by_pred.get('momentum', {}).get('stable', {}).get('strict', {})
    mom_means = mom_stable.get('mean', [])
    k_values = [5, 10, 25, 50, 75, 100]

    # Cascade highlights
    cascade = data.get('cascade_summary', {})
    cascade_lines = []
    for key, cdata in cascade.items():
        n = cdata.get('n_runs', 0)
        drifts = cdata.get('mean_l2_per_link', [])
        if drifts:
            cascade_lines.append(f'  {key}: final drift {drifts[-1]:.6f} (n={n})')

    summary_text = (
        f"Phase 2c: Aggressive Speculative Depth\n"
        f"{'=' * 40}\n\n"
        f"Seeds: {n_seeds}\n"
        f"Best predictor: {best_pred}\n"
        f"Max safe K (>50%): {max_safe_K}\n"
        f"Hypothetical speedup: {mean_speedup:.1%} +/- {std_speedup:.1%}\n\n"
        f"Momentum acceptance (stable, strict):\n"
    )

    for i, K in enumerate(k_values):
        if i < len(mom_means):
            summary_text += f'  K={K:>3}: {mom_means[i]:.1%}\n'

    if cascade_lines:
        summary_text += f'\nCascade results:\n'
        summary_text += '\n'.join(cascade_lines) + '\n'

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    return ax


# ============================================================
# Master Functions
# ============================================================

def plot_aggregate(data, save_dir=None, show=False):
    """Generate 6-panel figure from aggregate.json."""
    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    plot_depth_curve(data, ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    plot_cascade_chain(data, ax=ax2)

    ax3 = fig.add_subplot(gs[1, 1])
    plot_acceptance_comparison(data, ax=ax3)

    ax4 = fig.add_subplot(gs[2, :])
    plot_landscape_v2(data, ax=ax4)

    ax5 = fig.add_subplot(gs[3, 0])
    plot_cross_seed(data, ax=ax5)

    ax6 = fig.add_subplot(gs[3, 1])
    plot_summary_panel(data, ax=ax6)

    fig.suptitle('Phase 2c: Aggressive Speculative Depth (5-Seed Aggregate)',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase2c_aggregate.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase2c_aggregate.pdf', bbox_inches='tight')
        print(f"Phase 2c aggregate plots saved to {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_single_seed(data, save_dir=None, show=False):
    """Generate plots for a single seed's combined_results.json."""
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1])

    # Panel 1: Loss curve with regime bands
    ax1 = fig.add_subplot(gs[0, :])
    steps = data.get('loss_curve', {}).get('steps', [])
    losses = data.get('loss_curve', {}).get('values', [])
    regime_steps = data.get('regime_timeline', {}).get('steps', [])
    regimes = data.get('regime_timeline', {}).get('regimes', [])

    for i, regime in enumerate(regimes):
        x_start = regime_steps[i - 1] if i > 0 else 0
        x_end = regime_steps[i]
        color = REGIME_COLORS.get(regime, '#95a5a6')
        ax1.axvspan(x_start, x_end, color=color, alpha=0.15)

    ax1.plot(steps, losses, color='#2c3e50', linewidth=1.5)

    applied = [p for p in data.get('prediction_log', []) if p.get('actually_applied')]
    for pred in applied:
        s = pred['step']
        if s in steps:
            idx = steps.index(s)
            ax1.scatter(s, losses[idx], color='#9b59b6', marker='v', s=120, zorder=5,
                       edgecolors='white', linewidths=1)

    _style_ax(ax1, 'Training Step', 'Loss', f'Training Loss (seed {data.get("config", {}).get("seed", "?")})')

    # Panel 2: K-sweep bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    summary = data.get('depth_sweep', {}).get('summary_by_k', {})
    k_values = data.get('depth_sweep', {}).get('k_values', [5, 10, 25, 50, 75, 100])
    x = np.arange(len(k_values))
    width = 0.25
    for i, pred in enumerate(['momentum', 'linear', 'quadratic']):
        rates = []
        for K in k_values:
            k_data = summary.get(str(K), summary.get(K, {}))
            pred_data = k_data.get(pred, {})
            stable_data = pred_data.get('stable', {})
            rates.append(stable_data.get('strict_rate', 0) * 100)
        color = PREDICTOR_COLORS.get(pred, '#95a5a6')
        offset = (i - 1) * width
        ax2.bar(x + offset, rates, width, color=color, alpha=0.7,
                label=pred.capitalize(), edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'K={K}' for K in k_values], fontsize=9)
    ax2.set_ylim(0, 105)
    _style_ax(ax2, 'K', 'Strict Acceptance Rate (%)', 'K-Sweep (Stable Regime)')
    ax2.legend(fontsize=8)

    # Panel 3: Landscape V2
    ax3 = fig.add_subplot(gs[1, 1])
    probes = data.get('landscape_probe_v2', [])
    if probes:
        fractions = probes[0].get('fractions', [0.25, 0.5, 1.0, 2.0])
        for regime in ['stable', 'transition']:
            rp = [p for p in probes if p.get('regime') == regime]
            if rp:
                all_losses = []
                for p in rp:
                    origin = p.get('loss_at_origin', 0)
                    fl = p.get('losses_at_fractions', [])
                    if fl and origin:
                        all_losses.append([(l - origin) for l in fl])
                if all_losses:
                    arr = np.array(all_losses)
                    color = REGIME_COLORS.get(regime, '#95a5a6')
                    ax3.plot(fractions, arr.mean(axis=0), color=color, marker='o',
                            linewidth=2, label=f'{regime} (n={len(rp)})')
                    ax3.fill_between(fractions, arr.mean(0) - arr.std(0),
                                    arr.mean(0) + arr.std(0), color=color, alpha=0.15)
        ax3.axhline(y=0, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        _style_ax(ax3, 'Fraction of Prediction', 'Loss Change', 'Landscape V2')
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'No landscape data', ha='center', va='center',
                transform=ax3.transAxes)
        _style_ax(ax3, '', '', 'Landscape V2')

    # Panel 4: Cascade results
    ax4 = fig.add_subplot(gs[2, 0])
    cascades = data.get('cascade_results', {}).get('configurations', [])
    if cascades:
        for c in cascades:
            links = c.get('links', [])
            if links:
                key = f'{c["chain_length"]}x{c["K_per_link"]}'
                color = CASCADE_COLORS.get(key, '#95a5a6')
                link_nums = [l['link'] + 1 for l in links]
                drifts = [l['l2_drift'] for l in links]
                ax4.plot(link_nums, drifts, color=color, marker='o',
                        linewidth=2, label=f'{key} (step {c["checkpoint_step"]})', alpha=0.7)
        _style_ax(ax4, 'Link Number', 'L2 Drift', 'Cascade Drift')
        ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, 'No cascade data', ha='center', va='center',
                transform=ax4.transAxes)
        _style_ax(ax4, '', '', 'Cascade Drift')

    # Panel 5: Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    cfg = data.get('config', {})
    summ = data.get('summary', {})
    text = (
        f"Seed: {cfg.get('seed', '?')}\n"
        f"Steps trained: {summ.get('total_steps_trained', 0)}\n"
        f"Steps skipped: {summ.get('total_steps_skipped', 0)}\n"
        f"Final val loss: {summ.get('final_val_loss', 0):.4f}\n"
        f"Pass 1 time: {summ.get('pass1_time', 0):.0f}s\n"
        f"Pass 2 time: {summ.get('pass2_time', 0):.0f}s\n"
        f"Pass 3 time: {summ.get('pass3_time', 0):.0f}s\n"
    )
    ax5.text(0.05, 0.95, text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    seed_val = cfg.get('seed', '?')
    fig.suptitle(f'Phase 2c: Single Seed {seed_val}',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f'phase2c_seed{seed_val}.png', dpi=150, bbox_inches='tight')
        print(f"Single-seed plots saved to {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    agg_path = Path(__file__).parent.parent.parent / 'results' / 'phase2c' / 'aggregate.json'
    if agg_path.exists():
        with open(agg_path) as f:
            data = json.load(f)
        plot_aggregate(data, save_dir=str(agg_path.parent))
        print('Aggregate plots done.')
    else:
        print(f'No aggregate at {agg_path}. Run run_phase2c.py first.')

        # Try single seed
        for seed in [42, 43, 44, 45, 46]:
            seed_path = agg_path.parent / f'seed{seed}' / 'combined_results.json'
            if seed_path.exists():
                with open(seed_path) as f:
                    data = json.load(f)
                plot_single_seed(data, save_dir=str(seed_path.parent))
                print(f'Single-seed plot done for seed {seed}.')
