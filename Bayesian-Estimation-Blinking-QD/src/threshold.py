"""
Threshold Analysis (Section 2.3 of the paper)
===============================================

This module implements conventional threshold analysis for extracting switching
rates from blinking time traces. This serves as a comparison baseline for the
Bayesian inference methods.

Threshold Analysis Steps:
    1. Choose a threshold intensity I_threshold.
    2. Assign states: on if I_n > I_threshold, off otherwise.
    3. Identify contiguous on/off intervals and their durations.
    4. Histogram the durations.
    5. Fit exponential decay to the duration histograms.
    6. Extract switching rates from the exponential fits.

Threshold Selection Methods (from the paper's footnote 4):
    (i)   Minimum between two peaks of the intensity histogram (double-Poisson fit)
    (ii)  Two standard deviations above mean background counts
    (iii) Highest possible background count: where off-state Poisson dips below 1
    (iv)  Midpoint between two peaks in the counts histogram

Limitations:
    - Arbitrary choice of threshold significantly affects results
    - Choice of histogram binning for duration distributions adds another
      arbitrary parameter
    - No mechanism for estimating uncertainty in extracted rates
    - Fails for low signal-to-noise or fast switching

References:
    Geordy et al., New J. Phys. 21 (2019) 063001, Section 2.3
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import poisson
from typing import Tuple, List, Optional
from .cache import disk_cache


def find_threshold_methods(counts: np.ndarray, mu_est: float = None, lam_est: float = None) -> dict:
    """
    Compute thresholds using the four methods described in the paper.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon count time series.
    mu_est : float, optional
        Estimated background rate (for fitting methods).
    lam_est : float, optional
        Estimated fluorescence rate (for fitting methods).

    Returns
    -------
    thresholds : dict
        Keys: 'min_between_peaks', 'two_sigma', 'max_background', 'midpoint'
        Values: computed threshold for each method.
    """
    thresholds = {}

    # Build histogram of counts
    max_count = int(np.max(counts))
    bins = np.arange(0, max_count + 2) - 0.5
    hist, bin_edges = np.histogram(counts, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Method (i): Minimum between two peaks of a double-Poisson fit
    # Fit double Poisson to the histogram
    try:
        if mu_est is not None and lam_est is not None:
            # Use provided estimates as starting points
            mu_guess = mu_est
            total_guess = mu_est + lam_est
        else:
            # Estimate from data
            mu_guess = np.percentile(counts, 25)
            total_guess = np.percentile(counts, 75)

        def double_poisson(k, w, mu1, mu2):
            """Mixture of two Poissons."""
            return w * poisson.pmf(k.astype(int), mu1) + (1 - w) * poisson.pmf(k.astype(int), mu2)

        k_int = bin_centers.astype(int)
        hist_norm = hist / np.sum(hist)

        popt, _ = curve_fit(
            double_poisson, k_int, hist_norm,
            p0=[0.5, mu_guess, total_guess],
            bounds=([0, 0.1, 0.1], [1, max_count, max_count]),
            maxfev=5000
        )
        w_fit, mu1_fit, mu2_fit = popt

        # Find minimum between the two peaks
        k_range = np.arange(int(min(mu1_fit, mu2_fit)), int(max(mu1_fit, mu2_fit)) + 1)
        if len(k_range) > 0:
            vals = double_poisson(k_range, *popt)
            min_idx = np.argmin(vals)
            thresholds['min_between_peaks'] = k_range[min_idx]
        else:
            thresholds['min_between_peaks'] = (mu1_fit + mu2_fit) / 2
    except Exception:
        # Fallback: use median
        thresholds['min_between_peaks'] = np.median(counts)

    # Method (ii): Two standard deviations above mean background
    # Estimate background from lower portion of counts
    if mu_est is not None:
        bg_mean = mu_est
        bg_std = np.sqrt(mu_est)  # Poisson std = sqrt(mean)
    else:
        # Use lower quartile as background estimate
        lower_counts = counts[counts < np.median(counts)]
        if len(lower_counts) > 0:
            bg_mean = np.mean(lower_counts)
            bg_std = np.std(lower_counts)
        else:
            bg_mean = np.min(counts)
            bg_std = np.sqrt(bg_mean)

    thresholds['two_sigma'] = bg_mean + 2 * bg_std

    # Method (iii): Highest possible background count
    # Position where off-state Poisson (normalized to N time steps) dips below 1
    N = len(counts)
    if mu_est is not None:
        bg_rate = mu_est
    else:
        bg_rate = bg_mean

    # Find k where N * Poisson(k | bg_rate) < 1
    k_test = np.arange(0, max_count + 1)
    expected_counts = N * poisson.pmf(k_test, bg_rate)
    above_one = np.where(expected_counts >= 1)[0]
    if len(above_one) > 0:
        thresholds['max_background'] = above_one[-1] + 0.5
    else:
        thresholds['max_background'] = bg_rate + 3 * np.sqrt(bg_rate)

    # Method (iv): Midpoint between two peaks
    # Find peaks in histogram
    try:
        # Simple peak finding: look for the two highest local maxima
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist, distance=3, prominence=1)
        if len(peaks) >= 2:
            # Take the two most prominent peaks
            sorted_peaks = peaks[np.argsort(hist[peaks])][-2:]
            thresholds['midpoint'] = np.mean(bin_centers[sorted_peaks])
        else:
            # Fallback to double Poisson peaks
            thresholds['midpoint'] = (mu1_fit + mu2_fit) / 2 if 'mu1_fit' in dir() else np.median(counts)
    except Exception:
        thresholds['midpoint'] = np.median(counts)

    return thresholds


def extract_durations(counts: np.ndarray, threshold: float) -> Tuple[List[int], List[int]]:
    """
    Extract on and off interval durations using a threshold.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    threshold : float
        Threshold value for state assignment.

    Returns
    -------
    on_durations : list of int
        Durations (in time steps) of consecutive on periods.
    off_durations : list of int
        Durations of consecutive off periods.
    """
    # Assign states based on threshold
    states = (counts > threshold).astype(int)

    on_durations = []
    off_durations = []

    # Find contiguous intervals
    current_state = states[0]
    current_duration = 1

    for i in range(1, len(states)):
        if states[i] == current_state:
            current_duration += 1
        else:
            if current_state == 1:
                on_durations.append(current_duration)
            else:
                off_durations.append(current_duration)
            current_state = states[i]
            current_duration = 1

    # Don't forget the last interval
    if current_state == 1:
        on_durations.append(current_duration)
    else:
        off_durations.append(current_duration)

    return on_durations, off_durations


def fit_exponential_decay(durations: List[int], n_bins: int = 10) -> Tuple[float, float]:
    """
    Fit an exponential decay to duration histogram to extract switching rate.

    The probability of an on-duration of length t is:
        P(t) ∝ (1-β)^{t-1} · β ≈ β · exp(-β·t) for small β

    Similarly for off-durations with α.

    Parameters
    ----------
    durations : list of int
        List of interval durations.
    n_bins : int
        Number of bins for the duration histogram.

    Returns
    -------
    rate : float
        Extracted switching probability (α or β).
    rate_std : float
        Standard error of the fit.
    """
    if len(durations) < 3:
        return 0.0, np.inf

    durations = np.array(durations)

    # Create histogram
    bins = np.linspace(0.5, np.max(durations) + 0.5, n_bins + 1)
    hist, bin_edges = np.histogram(durations, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Remove empty bins for fitting
    mask = hist > 0
    if np.sum(mask) < 2:
        return 0.0, np.inf

    x_fit = bin_centers[mask]
    y_fit = hist[mask].astype(float)

    # Normalize
    y_fit /= np.sum(y_fit) * (bin_edges[1] - bin_edges[0])

    # Fit exponential: f(t) = A * exp(-rate * t)
    def exp_decay(t, A, rate):
        return A * np.exp(-rate * t)

    try:
        popt, pcov = curve_fit(
            exp_decay, x_fit, y_fit,
            p0=[np.max(y_fit), 1.0 / np.mean(durations)],
            bounds=([0, 0], [np.inf, 10]),
            maxfev=5000
        )
        rate = popt[1]
        rate_std = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else np.inf
    except Exception:
        # Fallback: use simple estimate
        rate = 1.0 / np.mean(durations)
        rate_std = rate / np.sqrt(len(durations))

    return rate, rate_std


def threshold_analysis(
    counts: np.ndarray,
    threshold: float,
    n_bins: int = 10
) -> dict:
    """
    Perform complete threshold analysis for a given threshold value.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    threshold : float
        Threshold for state assignment.
    n_bins : int
        Number of bins for duration histograms.

    Returns
    -------
    result : dict
        'alpha': extracted switch-on probability
        'beta': extracted switch-off probability
        'alpha_std': standard error of alpha
        'beta_std': standard error of beta
        'on_durations': list of on durations
        'off_durations': list of off durations
        'states': binary state assignment
    """
    on_durations, off_durations = extract_durations(counts, threshold)

    # Fit exponential to get switching rates
    # Off→On rate (alpha) from off durations
    alpha, alpha_std = fit_exponential_decay(off_durations, n_bins)

    # On→Off rate (beta) from on durations
    beta, beta_std = fit_exponential_decay(on_durations, n_bins)

    # Also convert to probabilities (rate ≈ probability for small rates)
    # For DTMC: probability = 1 - exp(-rate) ≈ rate for small rate

    states = (counts > threshold).astype(int)

    return {
        'alpha': alpha,
        'beta': beta,
        'alpha_std': alpha_std,
        'beta_std': beta_std,
        'on_durations': on_durations,
        'off_durations': off_durations,
        'states': states,
        'threshold': threshold,
        'n_bins': n_bins,
    }

@disk_cache()
def threshold_analysis_sweep(
    counts: np.ndarray,
    threshold_range: Tuple[float, float],
    n_thresholds: int = 10,
    bin_range: Tuple[int, int] = (5, 20),
) -> dict:
    """
    Perform threshold analysis over a range of thresholds and bin numbers.

    This replicates the analysis in Figure 6(c), showing how arbitrary
    choices of threshold and binning affect the extracted rates.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    threshold_range : tuple
        (min_threshold, max_threshold) to sweep over.
    n_thresholds : int
        Number of threshold values to test.
    bin_range : tuple
        (min_bins, max_bins) for duration histograms.

    Returns
    -------
    result : dict
        'thresholds': array of threshold values
        'bin_values': array of bin numbers
        'alphas': 2D array [threshold_idx, bin_idx] of extracted alphas
        'betas': 2D array [threshold_idx, bin_idx] of extracted betas
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
    bin_values = np.arange(bin_range[0], bin_range[1] + 1)

    alphas = np.zeros((n_thresholds, len(bin_values)))
    betas = np.zeros((n_thresholds, len(bin_values)))

    for i, thresh in enumerate(thresholds):
        for j, n_bins in enumerate(bin_values):
            result = threshold_analysis(counts, thresh, n_bins)
            alphas[i, j] = result['alpha']
            betas[i, j] = result['beta']

    return {
        'thresholds': thresholds,
        'bin_values': bin_values,
        'alphas': alphas,
        'betas': betas,
    }
