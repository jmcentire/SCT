"""
Publication-quality plotting for Phase 0: Regime Detection.

Generates four key plots:
1. Loss curve with regime overlay (colored bands)
2. Activation similarity over time
3. Loss curvature proxy over time
4. Correlation scatter: similarity vs curvature
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Import our types
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.regime.detect import RegimeResult, Regime


# Color scheme
REGIME_COLORS = {
    Regime.STABLE: '#2ecc71',      # Green
    Regime.CHAOTIC: '#e74c3c',     # Red
    Regime.TRANSITION: '#f39c12',  # Yellow/Orange
}

REGIME_ALPHAS = {
    Regime.STABLE: 0.15,
    Regime.CHAOTIC: 0.20,
    Regime.TRANSITION: 0.15,
}


def _style_ax(ax, xlabel, ylabel, title=None):
    """Apply consistent styling."""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _add_regime_bands(ax, result: RegimeResult):
    """Add colored vertical bands for regime classification."""
    all_steps = result.losses  # Use the full steps array for x limits
    steps = result.steps

    for i, label in enumerate(result.labels):
        # Band extends from previous step to this step
        if i == 0:
            x_start = 0
        else:
            x_start = steps[i - 1]
        x_end = steps[i]

        color = REGIME_COLORS[label.regime]
        alpha = REGIME_ALPHAS[label.regime]
        ax.axvspan(x_start, x_end, color=color, alpha=alpha)


def _regime_legend():
    """Create legend patches for regimes."""
    return [
        mpatches.Patch(color=REGIME_COLORS[Regime.STABLE], alpha=0.4, label='Stable'),
        mpatches.Patch(color=REGIME_COLORS[Regime.CHAOTIC], alpha=0.4, label='Chaotic'),
        mpatches.Patch(color=REGIME_COLORS[Regime.TRANSITION], alpha=0.4, label='Transition'),
    ]


def plot_loss_with_regimes(result: RegimeResult, ax=None, show_thresholds=False):
    """
    Plot loss curve with regime classification overlay.

    This is the money plot — the one that shows whether regime detection works.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    losses = result.losses
    # Steps for loss values: 0, interval, 2*interval, ...
    loss_steps = np.arange(len(losses))
    if len(result.steps) > 0:
        # Reconstruct step numbers for all loss values
        # losses[0] is step 0, losses[1] is first checkpoint, etc.
        interval = result.steps[0] if len(result.steps) > 0 else 1
        loss_steps = np.arange(len(losses)) * interval

    # Add regime bands
    _add_regime_bands(ax, result)

    # Plot loss (skip the inf at step 0 if present)
    valid_mask = np.isfinite(losses)
    ax.plot(loss_steps[valid_mask], losses[valid_mask],
            color='#2c3e50', linewidth=1.5, label='Training Loss')

    _style_ax(ax, 'Training Step', 'Loss', 'Training Loss with Regime Detection')
    ax.legend(handles=[ax.get_lines()[0]] + _regime_legend(),
              loc='upper right', fontsize=9, framealpha=0.9)

    return ax


def plot_similarity(result: RegimeResult, ax=None):
    """Plot activation similarity over time with regime bands and thresholds."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    _add_regime_bands(ax, result)

    ax.plot(result.steps, result.similarities,
            color='#2980b9', linewidth=1.5, label='Mean Cosine Similarity')

    # Threshold lines
    ax.axhline(result.thresholds['high'], color=REGIME_COLORS[Regime.STABLE],
               linestyle='--', alpha=0.7, linewidth=1, label=f"Stable threshold ({result.thresholds['high']:.3f})")
    ax.axhline(result.thresholds['low'], color=REGIME_COLORS[Regime.CHAOTIC],
               linestyle='--', alpha=0.7, linewidth=1, label=f"Chaotic threshold ({result.thresholds['low']:.3f})")

    _style_ax(ax, 'Training Step', 'Cosine Similarity',
              'Activation Similarity Between Consecutive Checkpoints')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


def plot_curvature(result: RegimeResult, ax=None):
    """Plot loss curvature proxy over time with regime bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    _add_regime_bands(ax, result)

    valid = ~np.isnan(result.curvatures)
    if valid.any():
        ax.plot(result.steps[valid], result.curvatures[valid],
                color='#8e44ad', linewidth=1.5, label='Loss Curvature (2nd diff)')
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    _style_ax(ax, 'Training Step', 'Curvature (L″)',
              'Loss Curvature Proxy')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


def plot_variance(result: RegimeResult, ax=None):
    """Plot activation variance over time with regime bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    _add_regime_bands(ax, result)

    ax.plot(result.steps, result.variances,
            color='#e67e22', linewidth=1.5, label='Activation Variance')

    _style_ax(ax, 'Training Step', 'Variance',
              'Activation Variance (Sliding Window)')
    ax.legend(fontsize=9, framealpha=0.9)

    return ax


def plot_correlation(result: RegimeResult, ax=None):
    """Scatter plot: similarity vs curvature, colored by regime."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    valid = ~np.isnan(result.curvatures)
    for i, label in enumerate(result.labels):
        if i < len(result.curvatures) and valid[i]:
            ax.scatter(
                result.similarities[i],
                result.curvatures[i],
                color=REGIME_COLORS[label.regime],
                s=30, alpha=0.7, edgecolors='white', linewidths=0.5
            )

    _style_ax(ax, 'Cosine Similarity', 'Loss Curvature',
              'Similarity vs. Curvature by Regime')
    ax.legend(handles=_regime_legend(), fontsize=9, framealpha=0.9)

    # Add correlation coefficient
    valid_sims = result.similarities[valid]
    valid_curvs = result.curvatures[valid]
    if len(valid_sims) > 2:
        corr = np.corrcoef(valid_sims, valid_curvs)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return ax


def plot_all(result: RegimeResult, save_dir: str | None = None, show: bool = True):
    """Generate all four Phase 0 plots in a single figure."""
    fig = plt.figure(figsize=(14, 16))

    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3,
                          height_ratios=[1.2, 1, 1, 1.2])

    # Row 1: Loss with regimes (full width)
    ax1 = fig.add_subplot(gs[0, :])
    plot_loss_with_regimes(result, ax=ax1)

    # Row 2: Similarity
    ax2 = fig.add_subplot(gs[1, :])
    plot_similarity(result, ax=ax2)

    # Row 3: Curvature and Variance
    ax3 = fig.add_subplot(gs[2, 0])
    plot_curvature(result, ax=ax3)
    ax3.set_title('Loss Curvature', fontsize=11, fontweight='bold')

    ax4 = fig.add_subplot(gs[2, 1])
    plot_variance(result, ax=ax4)
    ax4.set_title('Activation Variance', fontsize=11, fontweight='bold')

    # Row 4: Correlation scatter and summary
    ax5 = fig.add_subplot(gs[3, 0])
    plot_correlation(result, ax=ax5)

    # Summary text
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    summary = result.summary()
    ax6.text(0.1, 0.9, summary, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    fig.suptitle('Phase 0: Training Regime Detection',
                 fontsize=16, fontweight='bold', y=0.98)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / 'phase0_regimes.png', dpi=150, bbox_inches='tight')
        fig.savefig(save_path / 'phase0_regimes.pdf', bbox_inches='tight')
        print(f"Plots saved to {save_path}")

    if show:
        plt.show()

    return fig
