"""
Plotting Module for Reproducing Paper Figures
==============================================

This module provides visualization functions to reproduce all figures
from the paper:
    - Figure 1: Time traces showing different blinking scenarios
    - Figure 4: DTMC single-step inference (credible regions + marginals)
    - Figure 5: High switching probability inference
    - Figure 6: Comparison with threshold analysis
    - Figure 7: CTMC inference on fast-switching data
    - Figure 8: Accuracy of single-step model vs true CTMC rates
    - Figure 9: Multi-step inference error vs d
    - Figure 10: DTMC multi-step inference example
    - Figure 11: State inference with convergence

References:
    Geordy et al., New J. Phys. 21 (2019) 063001
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from scipy.stats import chi2
from typing import Optional, Tuple
import os


# Set default plotting style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def get_output_dir(base_dir: str = None) -> str:
    """Get or create the output directory for figures."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_credible_contours(posterior: np.ndarray, levels: list = None) -> list:
    """
    Compute contour levels corresponding to credible regions.

    Parameters
    ----------
    posterior : np.ndarray
        2D normalized posterior distribution.
    levels : list of float
        Credible region probabilities (e.g., [0.50, 0.90, 0.99]).

    Returns
    -------
    contour_levels : list of float
        Posterior density values at which to draw contours.
    """
    if levels is None:
        levels = [0.50, 0.90, 0.99]

    # Sort posterior values in descending order
    sorted_post = np.sort(posterior.ravel())[::-1]
    cumsum = np.cumsum(sorted_post)
    cumsum /= cumsum[-1]

    contour_levels = []
    for level in levels:
        idx = np.searchsorted(cumsum, level)
        if idx < len(sorted_post):
            contour_levels.append(sorted_post[idx])
        else:
            contour_levels.append(sorted_post[-1])

    return contour_levels


def plot_figure1(data: dict, save_path: str = None):
    """
    Plot Figure 1: Time traces showing different blinking scenarios.

    (a) Clear blinking with well-separated states + threshold line + zoom inset
    (b) Higher noise, same rates
    (c) High switching rate

    Parameters
    ----------
    data : dict
        Output from simulation.generate_figure1_data().
    save_path : str, optional
        Path to save the figure.
    """
    fig = plt.figure(figsize=(14, 10))
    # Use 4 rows: row 0 = panel(a), row 1 = zoom inset, rows 2,3 = panels (b),(c)
    gs = fig.add_gridspec(4, 2, width_ratios=[4, 1], height_ratios=[2, 1.2, 2, 2],
                          hspace=0.35, wspace=0.05)

    panels = [
        ('fig1a', '(a)', 0),
        ('fig1b', '(b)', 2),
        ('fig1c', '(c)', 3),
    ]

    ax_traces = {}
    for key, label, row in panels:
        ax_trace = fig.add_subplot(gs[row, 0])
        ax_hist = fig.add_subplot(gs[row, 1])

        counts = data[key]['counts']
        N = len(counts)
        time_k = np.arange(N) / 1000.0  # x-axis in units of x1000

        ax_trace.plot(time_k, counts, 'b-', linewidth=0.3, alpha=0.8)
        ax_trace.set_ylabel('Counts')
        ax_trace.text(0.02, 0.92, label, transform=ax_trace.transAxes,
                      fontsize=12, fontweight='bold', va='top')

        # Add threshold line to panel (a)
        if key == 'fig1a':
            params = data[key]['params']
            threshold = params['mu'] + params['lam'] / 2
            ax_trace.axhline(y=threshold, color='magenta', linestyle='--',
                             linewidth=1.5, alpha=0.8)
            ax_traces['fig1a'] = (ax_trace, counts, params, threshold)

        if key == 'fig1c':
            ax_trace.set_xlabel('Time step (×1000)')

        # Histogram on the right
        max_count = int(np.max(counts))
        bins = np.arange(0, max_count + 2) - 0.5
        ax_hist.hist(counts, bins=bins, orientation='horizontal',
                     color='steelblue', alpha=0.7)
        ax_hist.set_ylim(ax_trace.get_ylim())
        ax_hist.set_yticklabels([])
        if row == 0:
            ax_hist.set_title('Occurrences')
            # Add threshold arrow annotation on histogram
            if key == 'fig1a':
                ax_hist.axhline(y=threshold, color='magenta', linestyle='--',
                                linewidth=1.5, alpha=0.8)
                ax_hist.annotate('threshold', xy=(ax_hist.get_xlim()[1]*0.5, threshold),
                                 xytext=(ax_hist.get_xlim()[1]*0.6, threshold + 8),
                                 fontsize=9, color='magenta', fontweight='bold',
                                 arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5))

    # Add zoom inset for panel (a) - showing on/off durations
    if 'fig1a' in ax_traces:
        ax_zoom = fig.add_subplot(gs[1, :])
        _, counts_a, params_a, threshold_a = ax_traces['fig1a']
        N_a = len(counts_a)

        # Zoom into a region showing clear on/off transitions
        zoom_start = int(N_a * 0.885)
        zoom_end = int(N_a * 0.935)
        zoom_time = np.arange(zoom_start, zoom_end)
        zoom_counts = counts_a[zoom_start:zoom_end]

        ax_zoom.plot(zoom_time, zoom_counts, 'b-', linewidth=0.8, alpha=0.8)
        ax_zoom.axhline(y=threshold_a, color='magenta', linestyle='--',
                        linewidth=1.5, alpha=0.8)
        ax_zoom.set_ylabel('Counts')
        ax_zoom.set_xlim(zoom_start, zoom_end)

        # Find on and off periods for annotation arrows
        states_zoom = zoom_counts > threshold_a
        off_start = None
        on_start = None
        on_end = None
        off_end = None
        for i in range(1, len(states_zoom)):
            if not states_zoom[i] and states_zoom[i-1] and off_start is None:
                off_start = i + zoom_start
            if states_zoom[i] and not states_zoom[i-1] and off_start is not None and on_start is None:
                off_end = i + zoom_start
                on_start = i + zoom_start
            if not states_zoom[i] and states_zoom[i-1] and on_start is not None:
                on_end = i + zoom_start
                break

        if off_start is None or off_end is None or on_start is None or on_end is None:
            off_start = zoom_start + 5
            off_end = zoom_start + len(states_zoom) // 3
            on_start = off_end
            on_end = zoom_start + 2 * len(states_zoom) // 3

        # Draw off-period arrow (orange)
        y_arrow_off = threshold_a + 5
        ax_zoom.annotate('', xy=(off_start, y_arrow_off), xytext=(off_end, y_arrow_off),
                         arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
        ax_zoom.text((off_start + off_end) / 2, y_arrow_off + 2, 'off',
                     ha='center', va='bottom', color='darkorange', fontsize=10, fontstyle='italic')

        # Draw on-period arrow (orange)
        y_arrow_on = threshold_a - 8
        ax_zoom.annotate('', xy=(on_start, y_arrow_on), xytext=(on_end, y_arrow_on),
                         arrowprops=dict(arrowstyle='<->', color='darkorange', lw=2))
        ax_zoom.text((on_start + on_end) / 2, y_arrow_on - 3, 'on',
                     ha='center', va='top', color='darkorange', fontsize=10, fontstyle='italic')

        # Removed mark_inset to avoid enormous bounding box from tight_layout

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure 1 saved to: {save_path}")
    plt.close()


def plot_posterior_2d(
    posterior: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    true_x: float = None,
    true_y: float = None,
    xlabel: str = r'$\alpha_1$',
    ylabel: str = r'$\beta_1$',
    title: str = '',
    levels: list = None,
    ax: plt.Axes = None,
    save_path: str = None
):
    """
    Plot 2D posterior distribution with credible region contours.

    Parameters
    ----------
    posterior : np.ndarray
        2D posterior distribution (normalized).
    x_grid, y_grid : np.ndarray
        Grid values for x and y axes.
    true_x, true_y : float, optional
        True parameter values (shown as red dot).
    xlabel, ylabel : str
        Axis labels.
    title : str
        Plot title.
    levels : list
        Credible region levels (default: [0.50, 0.90, 0.99]).
    ax : matplotlib Axes, optional
        Axes to plot on.
    save_path : str, optional
        Path to save figure.
    """
    if levels is None:
        levels = [0.50, 0.90, 0.99]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        standalone = True
    else:
        standalone = False

    # Compute contour levels
    contour_levels = compute_credible_contours(posterior, levels)

    # Plot filled contours with light background
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    ax.contourf(X, Y, posterior, levels=50, cmap='Blues', alpha=0.4)
    # Paper-style colored contour lines: red (50%), magenta (90%), navy (99%)
    contour_colors = ['red', 'magenta', 'navy']
    cs = ax.contour(X, Y, posterior, levels=sorted(contour_levels),
                    colors=list(reversed(contour_colors)), linewidths=[2, 1.5, 1.2])
    # Label contours with credible region percentages
    level_labels = {sorted(contour_levels)[i]: f'{int(levels[len(levels)-1-i]*100)}%'
                    for i in range(len(contour_levels))}
    ax.clabel(cs, inline=True, fontsize=8, fmt=level_labels)

    # Mark true value
    if true_x is not None and true_y is not None:
        ax.plot(true_x, true_y, 'ro', markersize=8, label='True value')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if standalone and save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def plot_figure4(result: dict, true_alpha: float, true_beta: float,
                 true_lam: float = None, true_mu: float = None,
                 save_path: str = None):
    """
    Plot Figure 4: DTMC single-step inference with credible regions and marginals.

    Paper layout: (a) 2D contour plot (left), (b) 4 stacked marginal histograms (right)

    Parameters
    ----------
    result : dict
        Output from dtmc_single.compute_posterior_grid().
    true_alpha, true_beta : float
        True switching probabilities.
    true_lam, true_mu : float, optional
        True rates (for marginal plots).
    save_path : str, optional
        Path to save figure.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.3)

    # (a) 2D posterior with labeled contours
    ax_main = fig.add_subplot(gs[:, 0])
    plot_posterior_2d(
        result['posterior'], result['alpha_grid'], result['beta_grid'],
        true_x=true_alpha, true_y=true_beta,
        xlabel=r'$\alpha_1$', ylabel=r'$\beta_1$',
        title='(a)',
        ax=ax_main
    )

    # (b) Marginal histograms
    # Alpha marginal
    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_alpha.bar(result['alpha_grid'], result['marginal_alpha'],
                 width=np.diff(result['alpha_grid']).mean() * 0.9,
                 color='steelblue', alpha=0.8)
    if true_alpha is not None:
        ax_alpha.axvline(x=true_alpha, color='r', linewidth=2)
    ax_alpha.set_ylabel(r'$P(\alpha_1)$')
    ax_alpha.set_xlabel(r'$\alpha_1$')

    # Beta marginal
    ax_beta = fig.add_subplot(gs[1, 1])
    ax_beta.bar(result['beta_grid'], result['marginal_beta'],
                width=np.diff(result['beta_grid']).mean() * 0.9,
                color='steelblue', alpha=0.8)
    if true_beta is not None:
        ax_beta.axvline(x=true_beta, color='r', linewidth=2)
    ax_beta.set_ylabel(r'$P(\beta_1)$')
    ax_beta.set_xlabel(r'$\beta_1$')

    # Mu marginal
    ax_mu = fig.add_subplot(gs[2, 1])
    if 'marginal_mu' in result and result['marginal_mu'] is not None:
        ax_mu.bar(result['mu_grid'], result['marginal_mu'],
                  width=np.diff(result['mu_grid']).mean() * 0.9,
                  color='steelblue', alpha=0.8)
        if true_mu is not None:
            ax_mu.axvline(x=true_mu, color='r', linewidth=2)
    ax_mu.set_ylabel(r'$P(\mu)$')
    ax_mu.set_xlabel(r'$\mu$')

    # Lambda marginal
    ax_lam = fig.add_subplot(gs[3, 1])
    if 'marginal_lam' in result and result['marginal_lam'] is not None:
        ax_lam.bar(result['lam_grid'], result['marginal_lam'],
                   width=np.diff(result['lam_grid']).mean() * 0.9,
                   color='steelblue', alpha=0.8)
        if true_lam is not None:
            ax_lam.axvline(x=true_lam, color='r', linewidth=2)
    ax_lam.set_ylabel(r'$P(\lambda)$')
    ax_lam.set_xlabel(r'$\lambda$')

    fig.suptitle('(b)', x=0.75, y=0.98, fontsize=12)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure 4 saved to: {save_path}")
    plt.close()


def plot_figure5(result: dict, true_alpha: float, true_beta: float,
                 counts: np.ndarray = None, save_path: str = None):
    """
    Plot Figure 5: Inference with high switching probability.

    Paper layout: (a) time trace + count histogram, (b) full posterior with zoomed inset

    Parameters
    ----------
    result : dict
        Posterior result from dtmc_single.compute_posterior_grid().
    true_alpha, true_beta : float
        True switching probabilities (e.g., 0.8 and 0.9).
    counts : np.ndarray, optional
        Time trace data for subplot (a).
    save_path : str, optional
        Path to save figure.
    """
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], width_ratios=[3, 1],
                          hspace=0.3, wspace=0.05)

    # (a) Time trace
    ax_trace = fig.add_subplot(gs[0, 0])
    if counts is not None:
        ax_trace.plot(counts, 'b-', linewidth=0.3, alpha=0.8)
        ax_trace.set_xlabel('Time step')
        ax_trace.set_ylabel('Counts')
        ax_trace.set_title('(a)')

    # Count histogram (right of time trace)
    ax_hist = fig.add_subplot(gs[0, 1])
    if counts is not None:
        ax_hist.hist(counts, bins=20, orientation='horizontal',
                     color='steelblue', alpha=0.8)
        ax_hist.set_xlabel('Occurrences')
        ax_hist.set_ylim(ax_trace.get_ylim())
        ax_hist.set_yticklabels([])

    # (b) Full posterior with clean zoomed inset
    ax_post = fig.add_subplot(gs[1, :])

    # Compute contour levels
    contour_levels = compute_credible_contours(result['posterior'], [0.50, 0.90, 0.99])
    X, Y = np.meshgrid(result['alpha_grid'], result['beta_grid'], indexing='ij')

    # Main plot: only contour lines (no fill) for cleanliness on [0,1]x[0,1]
    ax_post.contour(X, Y, result['posterior'], levels=sorted(contour_levels),
                    colors=['navy', 'magenta', 'red'], linewidths=[1.2, 1.5, 2])
    if true_alpha is not None and true_beta is not None:
        ax_post.plot(true_alpha, true_beta, 'ro', markersize=6)
    ax_post.set_xlim(0, 1)
    ax_post.set_ylim(0, 1)
    ax_post.set_xlabel(r'$\alpha_1$')
    ax_post.set_ylabel(r'$\beta_1$')
    ax_post.set_title('(b)')

    # Add clean zoomed inset with only contour lines
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # Determine zoom region around MAP
    idx = np.unravel_index(np.argmax(result['posterior']), result['posterior'].shape)
    alpha_map = result['alpha_grid'][idx[0]]
    beta_map = result['beta_grid'][idx[1]]
    
    a_min, a_max = alpha_map - 0.08, alpha_map + 0.08
    b_min, b_max = beta_map - 0.06, beta_map + 0.06
    a_range = a_max - a_min
    b_range = b_max - b_min

    ax_inset = inset_axes(ax_post, width="40%", height="40%",
                          loc='center right', borderpad=2)

    # Inset: clean contour lines only, no fill
    cs_inset = ax_inset.contour(X, Y, result['posterior'], levels=sorted(contour_levels),
                                colors=['navy', 'magenta', 'red'], linewidths=[1.0, 1.2, 1.5])
    if true_alpha is not None and true_beta is not None:
        ax_inset.plot(true_alpha, true_beta, 'ro', markersize=5)

    # Set inset limits with some padding
    ax_inset.set_xlim(a_min - 0.02 * a_range, a_max + 0.02 * a_range)
    ax_inset.set_ylim(b_min - 0.02 * b_range, b_max + 0.02 * b_range)
    ax_inset.set_xlabel(r'$\alpha_1$', fontsize=8)
    ax_inset.set_ylabel(r'$\beta_1$', fontsize=8)
    ax_inset.tick_params(labelsize=7)

    # Add labeled contour percentages in inset
    level_labels = {sorted(contour_levels)[0]: '99%',
                    sorted(contour_levels)[1]: '90%',
                    sorted(contour_levels)[2]: '50%'}
    ax_inset.clabel(cs_inset, inline=True, fontsize=7, fmt=level_labels)

    # Connect inset to main plot region
    mark_inset(ax_post, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.8)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure 5 saved to: {save_path}")
    plt.close()


def plot_figure6(
    counts: np.ndarray,
    bayesian_result: dict,
    threshold_sweep: dict,
    true_alpha: float,
    true_beta: float,
    mu_est: float = None,
    lam_est: float = None,
    save_path: str = None
):
    """
    Plot Figure 6: Comparison of Bayesian inference with threshold analysis.

    Paper layout:
    (a) Top-left: Zoomed histogram of counts with threshold lines
    (b) Bottom-left: Stacked on-duration histograms with different binning
    (c) Right: Scatter plot of extracted rates vs Bayesian credible regions

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    bayesian_result : dict
        Posterior from Bayesian inference.
    threshold_sweep : dict
        Results from threshold_analysis_sweep.
    true_alpha, true_beta : float
        True switching probabilities.
    mu_est, lam_est : float, optional
        Estimated rates for threshold visualization.
    save_path : str, optional
        Path to save figure.
    """
    from .threshold import extract_durations

    fig = plt.figure(figsize=(12, 8))
    # Paper layout: (a)/(b) stacked on left, (c) on right
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.3)

    # (a) Zoomed histogram of overlap region with threshold lines
    ax1 = fig.add_subplot(gs[0, 0])
    if mu_est is not None and lam_est is not None:
        zoom_low = int(mu_est + (lam_est - mu_est) * 0.3)
        zoom_high = int(mu_est + (lam_est + mu_est) * 0.5)
    else:
        zoom_low = int(np.percentile(counts, 20))
        zoom_high = int(np.percentile(counts, 80))
    zoom_mask = (counts >= zoom_low) & (counts <= zoom_high)
    bins_zoom = np.arange(zoom_low, zoom_high + 2) - 0.5
    ax1.hist(counts[zoom_mask], bins=bins_zoom, color='steelblue', alpha=0.8)

    # Mark specific threshold values with colored lines
    thresholds_to_mark = threshold_sweep['thresholds']
    thresh_colors = ['magenta', 'goldenrod', 'cyan', 'darkorange',
                     'red', 'purple', 'brown', 'pink', 'gray', 'black']
    for i, thresh in enumerate(thresholds_to_mark):
        if zoom_low <= thresh <= zoom_high:
            ax1.axvline(x=thresh, color=thresh_colors[i % len(thresh_colors)],
                        linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Counts')
    ax1.set_ylabel('Occurrences')
    ax1.set_title('(a)')
    ax1.set_xlim(zoom_low - 0.5, zoom_high + 0.5)

    # (b) Stacked on-duration histograms for different binning choices
    # Paper style: separate small histograms stacked vertically (step/line style)
    ax2 = fig.add_subplot(gs[1, 0])
    mid_thresh = thresholds_to_mark[len(thresholds_to_mark) // 2]
    on_durs, off_durs = extract_durations(counts, mid_thresh)

    bin_choices = [8, 9, 10, 11, 12]
    colors_b = ['steelblue', 'orange', 'green', 'goldenrod', 'brown']
    n_stacked = len(bin_choices)

    if len(on_durs) > 0:
        max_dur = max(on_durs)
        # Stack histograms vertically with clear separation
        for k, (n_bins, col) in enumerate(zip(bin_choices, colors_b)):
            bins_dur = np.linspace(0, max_dur, n_bins + 1)
            hist_vals, bin_edges = np.histogram(on_durs, bins=bins_dur)
            # Use step plot style like the paper
            offset = (n_stacked - 1 - k) * (max(hist_vals) * 1.5 + 2)
            ax2.step(bin_edges[:-1], hist_vals + offset, where='post',
                     color=col, linewidth=1.2)
            ax2.step([bin_edges[-2], bin_edges[-1]], [hist_vals[-1] + offset, offset],
                     where='post', color=col, linewidth=1.2)
            # Add horizontal baseline
            ax2.axhline(y=offset, color=col, linewidth=0.3, alpha=0.3)
            ax2.text(max_dur * 0.75, offset + max(hist_vals) * 0.5,
                     f'{n_bins} bins', fontsize=8, color=col)

    ax2.set_xlabel('on duration (time steps)')
    ax2.set_ylabel('Occurrences')
    ax2.set_title('(b)')
    ax2.set_yticks([])

    # (c) Scatter plot colored by threshold, with Bayesian credible contours
    ax3 = fig.add_subplot(gs[:, 1])  # Span both rows on right side

    # Plot threshold analysis points colored by threshold value
    alphas = threshold_sweep['alphas']
    betas = threshold_sweep['betas']
    thresholds_arr = threshold_sweep['thresholds']

    cmap = plt.cm.tab10
    markers_list = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    for i, thresh in enumerate(thresholds_arr):
        color = cmap(i % 10)
        marker = markers_list[i % len(markers_list)]
        ax3.scatter(alphas[i, :], betas[i, :], c=[color], s=25, alpha=0.7,
                    marker=marker, label=f'{int(thresh)}')

    # Plot Bayesian credible regions
    contour_levels = compute_credible_contours(bayesian_result['posterior'], [0.50, 0.90, 0.99])
    X, Y = np.meshgrid(bayesian_result['alpha_grid'], bayesian_result['beta_grid'], indexing='ij')
    ax3.contour(X, Y, bayesian_result['posterior'],
                levels=sorted(contour_levels),
                colors=['navy', 'magenta', 'red'], linewidths=[2, 1.5, 1.2])

    # True value
    ax3.plot(true_alpha, true_beta, 'ro', markersize=10, zorder=10)

    # Mean and std of "reasonable" threshold range
    reasonable_mask = (thresholds_arr >= zoom_low + 2) & (thresholds_arr <= zoom_high - 2)
    if np.any(reasonable_mask):
        mean_a = np.mean(alphas[reasonable_mask, :])
        mean_b = np.mean(betas[reasonable_mask, :])
        std_a = np.std(alphas[reasonable_mask, :])
        std_b = np.std(betas[reasonable_mask, :])
        ax3.errorbar(mean_a, mean_b, xerr=std_a, yerr=std_b,
                     fmt='s', color='gray', markersize=8, capsize=3, zorder=5)

    ax3.set_xlabel(r'$\alpha_1$')
    ax3.set_ylabel(r'$\beta_1$')
    ax3.set_title('(c)')
    ax3.legend(title='Threshold', fontsize=7, loc='upper right', ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure 6 saved to: {save_path}")
    plt.close()


def plot_figure7(result: dict, true_ra: float, true_rb: float,
                 true_lam: float = None, true_mu: float = None,
                 save_path: str = None):
    """
    Plot Figure 7: CTMC inference on fast-switching data.

    Paper layout: (a) contour on left, (b) 4 stacked marginal histograms on right

    Parameters
    ----------
    result : dict
        Output from ctmc.compute_posterior_ctmc().
    true_ra, true_rb : float
        True switching rates.
    true_lam, true_mu : float, optional
        True emission rates.
    save_path : str, optional
        Path to save figure.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.3)

    # (a) 2D posterior with labeled contours
    ax_main = fig.add_subplot(gs[:, 0])
    plot_posterior_2d(
        result['posterior'], result['ra_grid'], result['rb_grid'],
        true_x=true_ra, true_y=true_rb,
        xlabel=r'$r_a$', ylabel=r'$r_b$',
        title='(a)',
        ax=ax_main
    )

    # (b) Marginal histograms
    # r_a marginal
    ax_ra = fig.add_subplot(gs[0, 1])
    ax_ra.bar(result['ra_grid'], result['marginal_ra'],
              width=np.diff(result['ra_grid']).mean() * 0.9,
              color='steelblue', alpha=0.8)
    if true_ra is not None:
        ax_ra.axvline(x=true_ra, color='r', linewidth=2)
    ax_ra.set_ylabel(r'$P(r_a)$')
    ax_ra.set_xlabel(r'$r_a$')

    # r_b marginal
    ax_rb = fig.add_subplot(gs[1, 1])
    ax_rb.bar(result['rb_grid'], result['marginal_rb'],
              width=np.diff(result['rb_grid']).mean() * 0.9,
              color='steelblue', alpha=0.8)
    if true_rb is not None:
        ax_rb.axvline(x=true_rb, color='r', linewidth=2)
    ax_rb.set_ylabel(r'$P(r_b)$')
    ax_rb.set_xlabel(r'$r_b$')

    # Mu marginal
    ax_mu = fig.add_subplot(gs[2, 1])
    if 'marginal_mu' in result and result['marginal_mu'] is not None:
        ax_mu.bar(result['mu_grid'], result['marginal_mu'],
                  width=np.diff(result['mu_grid']).mean() * 0.9,
                  color='steelblue', alpha=0.8)
        if true_mu is not None:
            ax_mu.axvline(x=true_mu, color='r', linewidth=2)
    ax_mu.set_ylabel(r'$P(\mu)$')
    ax_mu.set_xlabel(r'$\mu$')

    # Lambda marginal
    ax_lam = fig.add_subplot(gs[3, 1])
    if 'marginal_lam' in result and result['marginal_lam'] is not None:
        ax_lam.bar(result['lam_grid'], result['marginal_lam'],
                   width=np.diff(result['lam_grid']).mean() * 0.9,
                   color='steelblue', alpha=0.8)
        if true_lam is not None:
            ax_lam.axvline(x=true_lam, color='r', linewidth=2)
    ax_lam.set_ylabel(r'$P(\lambda)$')
    ax_lam.set_xlabel(r'$\lambda$')

    fig.suptitle('(b)', x=0.75, y=0.98, fontsize=12)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure 7 saved to: {save_path}")
    plt.close()


def plot_figure8(error_map: np.ndarray, ra_values: np.ndarray, rb_values: np.ndarray,
                 example_posteriors: list = None, save_path: str = None):
    """
    Plot Figure 8: Accuracy of single-step inference for CTMC-generated data.

    Color map of Euclidean error between true rates and posterior mode.

    Parameters
    ----------
    error_map : np.ndarray
        2D array of Euclidean errors.
    ra_values, rb_values : np.ndarray
        Rate values used for the grid.
    example_posteriors : list, optional
        List of (result_dict, true_ra, true_rb) for inset plots.
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Color map
    im = ax.pcolormesh(ra_values, rb_values, error_map.T,
                       cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label='Euclidean error')

    # Threshold line: r_a * r_b = 0.1
    ra_line = np.linspace(0.01, ra_values[-1], 100)
    rb_line = 0.1 / ra_line
    mask = rb_line <= rb_values[-1]
    ax.plot(ra_line[mask], rb_line[mask], 'c-', linewidth=2,
            label=r'$r_a \cdot r_b = 0.1$')

    ax.set_xlabel(r'$r_a$')
    ax.set_ylabel(r'$r_b$')
    ax.set_title('Figure 8: Error of single-step model on CTMC data')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figure 8 saved to: {save_path}")
    plt.close()


def plot_figure9(rate_values, d_values: list, errors_dict: dict,
                 inset_data: dict = None, save_path: str = None):
    """
    Plot Figure 9: Multi-step inference error vs switching rate for various d.

    Paper format: x-axis is r_a = r_b, y-axis is Euclidean error,
    multiple dashed lines for each d value, shaded 25% relative error region.
    Includes inset with posterior contours and diagonal line labels.

    Parameters
    ----------
    rate_values : np.ndarray
        Switching rate values tested (x-axis).
    d_values : list
        Values of d tested (one line per d).
    errors_dict : dict
        Keys are d values, values are lists of errors for each rate.
    inset_data : dict, optional
        Data for inset contour plot. Keys: 'rate', 'posteriors' (dict of d -> result).
    save_path : str, optional
        Path to save figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(d_values)))

    for idx, d in enumerate(d_values):
        errors = errors_dict[d]
        ax.plot(rate_values[:len(errors)], errors,
                marker=markers[idx % len(markers)],
                color=colors[idx],
                linewidth=1.5, markersize=5,
                linestyle='--',
                label=f'$d = {d}$')

        # Add diagonal label on the line (paper style)
        if len(errors) > 3:
            label_idx = int(len(errors) * 0.6)
            if label_idx < len(errors):
                x_label = rate_values[label_idx]
                y_label = errors[label_idx]
                ax.annotate(f'd = {d}', xy=(x_label, y_label),
                           fontsize=8, color=colors[idx], fontstyle='italic',
                           ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')

    # 25% relative error shaded region: error < 0.25 * sqrt(2) * rate
    rate_line = np.linspace(0, rate_values[-1] * 1.05, 100)
    threshold_line = 0.25 * rate_line * np.sqrt(2)
    ax.fill_between(rate_line, 0, threshold_line, alpha=0.15, color='gray')
    ax.plot(rate_line, threshold_line, 'k--', linewidth=0.8, alpha=0.5)

    ax.set_xlabel(r'$r_a = r_b$')
    ax.set_ylabel('Error (Euclidean dist.)')
    ax.legend(fontsize=9)
    ax.set_xlim(0, rate_values[-1] * 1.05)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Add inset with posterior contours if data is provided
    if inset_data is not None and 'posteriors' in inset_data:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="35%", height="35%", loc='upper left',
                              borderpad=3)
        inset_colors_map = {2: 'green', 4: 'teal', 8: 'orange'}
        for d_val, (post, ra_g, rb_g) in inset_data['posteriors'].items():
            if d_val == 1:
                continue  # Skip d=1 in inset
            cl = compute_credible_contours(post, [0.50, 0.90, 0.99])
            Xi, Yi = np.meshgrid(ra_g, rb_g, indexing='ij')
            col = inset_colors_map.get(d_val, 'gray')
            # Ensure levels are unique and strictly increasing
            cl_sorted = sorted(set(cl))
            if len(cl_sorted) >= 2:
                ax_inset.contour(Xi, Yi, post, levels=cl_sorted,
                                colors=[col], linewidths=[1.0, 0.8, 0.6][:len(cl_sorted)])

        # Mark true value
        true_rate = inset_data['rate']
        ax_inset.plot(true_rate, true_rate, 'ro', markersize=4)
        ax_inset.set_xlabel(r'$r_a$', fontsize=7)
        ax_inset.set_ylabel(r'$r_b$', fontsize=7)
        ax_inset.tick_params(labelsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Figure 9 saved to: {save_path}")
    plt.close()


def plot_figure10(result: dict, true_ra: float, true_rb: float,
                  d: int = 16, true_lam: float = None, true_mu: float = None,
                  save_path: str = None):
    """
    Plot Figure 10: DTMC multi-step inference example.

    Paper layout: (a) contour on left, (b) 4 stacked marginal histograms on right

    Parameters
    ----------
    result : dict
        Output from dtmc_multi.compute_posterior_multi().
    true_ra, true_rb : float
        True switching rates.
    d : int
        Number of subintervals used.
    true_lam, true_mu : float, optional
        True emission rates.
    save_path : str, optional
        Path to save figure.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.4, wspace=0.3)

    # (a) 2D posterior
    ax_main = fig.add_subplot(gs[:, 0])
    plot_posterior_2d(
        result['posterior'], result['ra_grid'], result['rb_grid'],
        true_x=true_ra, true_y=true_rb,
        xlabel=r'$r_a$', ylabel=r'$r_b$',
        title=f'(a) DTMC multi-step d={d}',
        ax=ax_main
    )

    # (b) Marginal histograms
    ax_ra = fig.add_subplot(gs[0, 1])
    ax_ra.bar(result['ra_grid'], result['marginal_ra'],
              width=np.diff(result['ra_grid']).mean() * 0.9,
              color='steelblue', alpha=0.8)
    if true_ra is not None:
        ax_ra.axvline(x=true_ra, color='r', linewidth=2)
    ax_ra.set_ylabel(r'$P(r_a)$')
    ax_ra.set_xlabel(r'$r_a$')

    ax_rb = fig.add_subplot(gs[1, 1])
    ax_rb.bar(result['rb_grid'], result['marginal_rb'],
              width=np.diff(result['rb_grid']).mean() * 0.9,
              color='steelblue', alpha=0.8)
    if true_rb is not None:
        ax_rb.axvline(x=true_rb, color='r', linewidth=2)
    ax_rb.set_ylabel(r'$P(r_b)$')
    ax_rb.set_xlabel(r'$r_b$')

    ax_mu = fig.add_subplot(gs[2, 1])
    if 'marginal_mu' in result and result['marginal_mu'] is not None:
        ax_mu.bar(result['mu_grid'], result['marginal_mu'],
                  width=np.diff(result['mu_grid']).mean() * 0.9,
                  color='steelblue', alpha=0.8)
        if true_mu is not None:
            ax_mu.axvline(x=true_mu, color='r', linewidth=2)
    ax_mu.set_ylabel(r'$P(\mu)$')
    ax_mu.set_xlabel(r'$\mu$')

    ax_lam = fig.add_subplot(gs[3, 1])
    if 'marginal_lam' in result and result['marginal_lam'] is not None:
        ax_lam.bar(result['lam_grid'], result['marginal_lam'],
                   width=np.diff(result['lam_grid']).mean() * 0.9,
                   color='steelblue', alpha=0.8)
        if true_lam is not None:
            ax_lam.axvline(x=true_lam, color='r', linewidth=2)
    ax_lam.set_ylabel(r'$P(\lambda)$')
    ax_lam.set_xlabel(r'$\lambda$')

    fig.suptitle('(b)', x=0.75, y=0.98, fontsize=12)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure 10 saved to: {save_path}")
    plt.close()


def plot_figure11(
    counts: np.ndarray,
    true_states: np.ndarray,
    state_probs: np.ndarray,
    threshold_states: np.ndarray,
    posterior_result: dict = None,
    true_alpha: float = None,
    true_beta: float = None,
    save_path: str = None
):
    """
    Plot Figure 11: State inference demonstration.

    (a) Simulated count data with threshold
    (b) Threshold-determined states
    (c) True states from simulation
    (d) Posterior probability of state=1
    (e) Posterior of alpha, beta (optional)

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    true_states : np.ndarray
        Ground truth states from simulation.
    state_probs : np.ndarray of shape (N, 2)
        Inferred state probabilities.
    threshold_states : np.ndarray
        Binary states from threshold analysis.
    posterior_result : dict, optional
        Posterior of alpha, beta for subplot (e).
    true_alpha, true_beta : float, optional
        True switching probabilities.
    save_path : str, optional
        Path to save figure.
    """
    N = len(counts)
    time_arr = np.arange(N)

    if posterior_result is not None:
        # Paper layout: (a)-(d) left 2/3, (e) right 1/3
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        gs = fig.add_gridspec(4, 2, width_ratios=[2, 1], hspace=0.08, wspace=0.15)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[2, 0])
        ax_d = fig.add_subplot(gs[3, 0])
        ax_e = fig.add_subplot(gs[:, 1])
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        ax_a, ax_b, ax_c, ax_d = axes
        ax_e = None

    # (a) Count data with threshold line (magenta dashed, matching paper)
    ax_a.plot(time_arr, counts, 'b-', linewidth=0.5, alpha=0.8)
    # Compute threshold as midpoint between off and on means
    threshold = np.mean(counts)  # fallback
    if threshold_states is not None:
        on_mask = threshold_states == 1
        off_mask = threshold_states == 0
        if np.any(on_mask) and np.any(off_mask):
            threshold = (np.mean(counts[on_mask]) + np.mean(counts[off_mask])) / 2
    ax_a.axhline(y=threshold, color='magenta', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_a.set_ylabel('Counts')
    ax_a.set_title('(a)')

    # (b) Threshold-determined states (orange, matching paper)
    ax_b.step(time_arr, threshold_states, color='darkorange', linewidth=0.8, where='mid')
    ax_b.set_ylabel('State')
    ax_b.set_ylim(-0.1, 1.1)
    ax_b.set_yticks([0, 1])
    ax_b.set_title('(b)')

    # (c) True states
    ax_c.step(time_arr, true_states, 'g-', linewidth=0.8, where='mid')
    ax_c.set_ylabel('State')
    ax_c.set_ylim(-0.1, 1.1)
    ax_c.set_yticks([0, 1])
    ax_c.set_title('(c)')

    # (d) Inferred P(state=1)
    ax_d.plot(time_arr, state_probs[:, 1], 'b-', linewidth=0.8)
    ax_d.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax_d.set_ylabel(r'$P(s_k=1|c)$')
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_xlabel('Time step')
    ax_d.set_title('(d)')

    # (e) Posterior of alpha, beta
    if ax_e is not None and posterior_result is not None:
        plot_posterior_2d(
            posterior_result['posterior'],
            posterior_result['alpha_grid'],
            posterior_result['beta_grid'],
            true_x=true_alpha, true_y=true_beta,
            xlabel=r'$\alpha_1$', ylabel=r'$\beta_1$',
            title='(e)',
            ax=ax_e
        )

    if save_path:
        plt.savefig(save_path)
        print(f"Figure 11 saved to: {save_path}")
    plt.close()
