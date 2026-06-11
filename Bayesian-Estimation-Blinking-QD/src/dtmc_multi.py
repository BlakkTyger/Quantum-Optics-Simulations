"""
DTMC Multi-Step Bayesian Inference (Section 4 of the paper)
============================================================

This module implements the DTMC multi-step model, which approximates the CTMC
by subdividing each detector interval into d = 2^m subintervals. The emitter
can switch state only at subinterval boundaries.

Key Idea:
    The recursion relation (Eq. 52) allows efficient computation of the count
    distribution over a full interval from knowledge of the distribution over
    half-intervals. By choosing d = 2^m subintervals, we apply this recursion
    m times starting from the base case (single subinterval with Poisson counts).

Recursion Relation (Eq. 52):
    f_{i,j}(τ, k) = Σ_{c_d=0}^{k} f_{i,0}(τ/2, c_d) · f_{0,j}(τ/2, k-c_d)
                   + Σ_{c_d=0}^{k} f_{i,1}(τ/2, c_d) · f_{1,j}(τ/2, k-c_d)

    This is the sum of two discrete convolutions over the intermediate state.

Base Case (Eq. in Section 4.2):
    f_{i,j}(1/d, k) = transition_prob(i→j) · Poisson(k | rate_i / d)

    where:
        f_{0,0}(1/d, k) = (1 - α_d) · Poisson(k | μ/d)
        f_{0,1}(1/d, k) = α_d · Poisson(k | μ/d)
        f_{1,0}(1/d, k) = β_d · Poisson(k | (μ+λ)/d)
        f_{1,1}(1/d, k) = (1 - β_d) · Poisson(k | (μ+λ)/d)

    and:
        α_d = 1 - exp(-r_a / d)    [Eq. 43]
        β_d = 1 - exp(-r_b / d)    [Eq. 44]

Connection to CTMC:
    As d → ∞, this model converges to the exact CTMC result. In practice,
    d = 2^m where m is chosen such that r_a · r_b < 0.1 × 2^{2m}.

References:
    Geordy et al., New J. Phys. 21 (2019) 063001, Section 4
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional
from tqdm import tqdm
from .cache import disk_cache


def compute_base_distributions(
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    d: int,
    max_count: int
) -> np.ndarray:
    """
    Compute the base case distributions f_{i,j}(1/d, k) for all counts k.

    These are the probability distributions for observing k counts in a single
    subinterval of duration 1/d, given start state i and end state j.

    Parameters
    ----------
    r_a : float
        Switch-on rate (per detector interval).
    r_b : float
        Switch-off rate (per detector interval).
    lam : float
        Fluorescence rate (per detector interval).
    mu : float
        Background rate (per detector interval).
    d : int
        Number of subintervals per detector interval.
    max_count : int
        Maximum count value to consider.

    Returns
    -------
    f_base : np.ndarray of shape (2, 2, max_count+1)
        f_base[i, j, k] = f_{i,j}(1/d, k)
        i = start state, j = end state, k = count value.
    """
    # Switching probabilities per subinterval (Eqs. 43, 44)
    alpha_d = 1.0 - np.exp(-r_a / d)
    beta_d = 1.0 - np.exp(-r_b / d)

    # Poisson rates per subinterval
    mu_sub = mu / d
    lam_sub = lam / d

    # Count range
    k_vals = np.arange(max_count + 1)

    # Compute Poisson probabilities
    p_off = poisson.pmf(k_vals, mu_sub)         # Counts when in off state
    p_on = poisson.pmf(k_vals, mu_sub + lam_sub)  # Counts when in on state

    # Base distributions: f[i, j, k] = P(transition i→j) × P(counts=k | state_i)
    # Note: counts depend on the state DURING the subinterval (= start state i)
    f_base = np.zeros((2, 2, max_count + 1))

    f_base[0, 0, :] = (1 - alpha_d) * p_off   # off→off, observe off counts
    f_base[0, 1, :] = alpha_d * p_off           # off→on, observe off counts
    f_base[1, 0, :] = beta_d * p_on             # on→off, observe on counts
    f_base[1, 1, :] = (1 - beta_d) * p_on       # on→on, observe on counts

    return f_base


def apply_recursion(f_half: np.ndarray, max_count: int) -> np.ndarray:
    """
    Apply the recursion relation (Eq. 52) to double the time interval.

    Given f_{i,j}(τ/2, k), compute f_{i,j}(τ, k) using discrete convolution
    over the intermediate state.

    f_{i,j}(τ, k) = Σ_{c=0}^{k} [f_{i,0}(τ/2, c) · f_{0,j}(τ/2, k-c)
                                  + f_{i,1}(τ/2, c) · f_{1,j}(τ/2, k-c)]

    Parameters
    ----------
    f_half : np.ndarray of shape (2, 2, K)
        Count distributions for half-interval: f_half[i, j, k].
    max_count : int
        Maximum count to retain in the convolution.

    Returns
    -------
    f_full : np.ndarray of shape (2, 2, max_count+1)
        Count distributions for the doubled interval.

    Notes
    -----
    The convolution is implemented using numpy's convolve for efficiency.
    The result is truncated to max_count+1 entries.
    """
    f_full = np.zeros((2, 2, max_count + 1))

    for i in range(2):      # start state
        for j in range(2):  # end state
            # Sum over intermediate state z (0 or 1)
            for z in range(2):
                # Convolve f_{i,z}(τ/2, ·) with f_{z,j}(τ/2, ·)
                conv = np.convolve(f_half[i, z, :], f_half[z, j, :])
                # Truncate to max_count+1
                f_full[i, j, :] += conv[:max_count + 1]

    return f_full


def compute_full_interval_distributions(
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    d: int,
    max_count: int
) -> np.ndarray:
    """
    Compute f_{i,j}(1, k) for the full detector interval using the recursion.

    Starting from the base case at 1/d, applies the recursion m = log2(d) times
    to build up to the full interval.

    Parameters
    ----------
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.
    lam : float
        Fluorescence rate.
    mu : float
        Background rate.
    d : int
        Number of subintervals (must be power of 2).
    max_count : int
        Maximum count to consider.

    Returns
    -------
    f_full : np.ndarray of shape (2, 2, max_count+1)
        f_full[i, j, k] = P(counts=k, end_state=j | start_state=i)
        for the full detector interval.
    """
    assert d > 0 and (d & (d - 1)) == 0, "d must be a power of 2"

    m = int(np.log2(d))

    # Base case
    f = compute_base_distributions(r_a, r_b, lam, mu, d, max_count)

    # Apply recursion m times to go from 1/d to 1
    for _ in range(m):
        f = apply_recursion(f, max_count)

    return f


def compute_transfer_matrix_multi(
    c_n: int,
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    d: int,
    max_count: int,
    f_full: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the transfer matrix R_n for one time step in the multi-step model.

    R_n[s_n, s_{n-1}] = P(c_n, s_n | s_{n-1}, Ω_d) = f_{s_{n-1}, s_n}(1, c_n)

    Parameters
    ----------
    c_n : int
        Observed count for this interval.
    r_a, r_b : float
        Switching rates.
    lam, mu : float
        Fluorescence and background rates.
    d : int
        Number of subintervals.
    max_count : int
        Maximum count considered.
    f_full : np.ndarray, optional
        Pre-computed full interval distributions. If None, computed here.

    Returns
    -------
    R_n : np.ndarray of shape (2, 2)
        Transfer matrix.
    """
    if f_full is None:
        f_full = compute_full_interval_distributions(r_a, r_b, lam, mu, d, max_count)

    R_n = np.zeros((2, 2))
    if c_n <= max_count:
        # R_n[new_state, prev_state] = f_full[prev_state, new_state, count]
        for s_next in range(2):
            for s_prev in range(2):
                R_n[s_next, s_prev] = f_full[s_prev, s_next, c_n]
    # If c_n > max_count, probability is essentially zero

    return R_n


def compute_log_likelihood_multi(
    counts: np.ndarray,
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    d: int = 8
) -> float:
    """
    Compute log P(c | Ω_d) for the DTMC multi-step model (Eq. 53).

    P(c | Ω_d) = [1,1] · ∏_{n=1}^{N} R_n · D_0

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.
    lam : float
        Fluorescence rate.
    mu : float
        Background rate.
    d : int
        Number of subintervals (power of 2).

    Returns
    -------
    log_likelihood : float
        Log marginal likelihood.
    """
    N = len(counts)
    max_count = int(np.max(counts)) + 10  # Padding for safety

    # Pre-compute f_full (same for all intervals given fixed parameters)
    f_full = compute_full_interval_distributions(r_a, r_b, lam, mu, d, max_count)

    # Initial state distribution (stationary)
    total_rate = r_a + r_b
    if total_rate > 0:
        D0 = np.array([r_b / total_rate, r_a / total_rate])
    else:
        D0 = np.array([0.5, 0.5])

    # Iterative matrix-vector product
    v = D0.copy()
    log_scale = 0.0

    for n in range(N):
        R_n = compute_transfer_matrix_multi(counts[n], r_a, r_b, lam, mu, d, max_count, f_full)
        v = R_n @ v

        scale = np.sum(v)
        if scale > 0:
            v /= scale
            log_scale += np.log(scale)
        else:
            return -np.inf

    result = np.sum(v)
    if result > 0:
        return log_scale + np.log(result)
    else:
        return -np.inf

@disk_cache()
def compute_posterior_multi(
    counts: np.ndarray,
    d: int = 8,
    ra_range: Tuple[float, float] = (0.01, 8.0),
    rb_range: Tuple[float, float] = (0.01, 8.0),
    lam_range: Optional[Tuple[float, float]] = None,
    mu_range: Optional[Tuple[float, float]] = None,
    lam_known: Optional[float] = None,
    mu_known: Optional[float] = None,
    n_ra: int = 70,
    n_rb: int = 70,
    n_lam: int = 50,
    n_mu: int = 50,
    show_progress: bool = True
) -> dict:
    """
    Compute posterior P(r_a, r_b | c, I_d) for the DTMC multi-step model.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    d : int
        Number of subintervals (power of 2).
    ra_range, rb_range : tuple
        Range for rate parameters.
    lam_range, mu_range : tuple, optional
        Range for marginalization (if rates unknown).
    lam_known, mu_known : float, optional
        Known rates (skips marginalization).
    n_ra, n_rb : int
        Grid resolution for switching rates.
    n_lam, n_mu : int
        Grid resolution for fluorescence/background.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    result : dict
        'posterior', 'ra_grid', 'rb_grid', 'marginal_ra', 'marginal_rb', etc.
    """
    ra_grid = np.linspace(ra_range[0], ra_range[1], n_ra)
    rb_grid = np.linspace(rb_range[0], rb_range[1], n_rb)

    marginalize_lam = lam_known is None
    marginalize_mu = mu_known is None

    if marginalize_lam:
        if lam_range is None:
            raise ValueError("lam_range required when lam_known is None")
        lam_grid = np.linspace(lam_range[0] + 0.1, lam_range[1], n_lam)
    else:
        lam_grid = np.array([lam_known])
        n_lam = 1

    if marginalize_mu:
        if mu_range is None:
            raise ValueError("mu_range required when mu_known is None")
        mu_grid = np.linspace(mu_range[0] + 0.1, mu_range[1], n_mu)
    else:
        mu_grid = np.array([mu_known])
        n_mu = 1

    log_posterior = np.full((n_ra, n_rb), -np.inf)

    iterator = range(n_ra)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Computing posterior (DTMC d={d})")

    for i in iterator:
        for j in range(n_rb):
            log_likes = np.zeros((n_lam, n_mu))
            for il in range(n_lam):
                for im in range(n_mu):
                    log_likes[il, im] = compute_log_likelihood_multi(
                        counts, ra_grid[i], rb_grid[j],
                        lam_grid[il], mu_grid[im], d
                    )

            max_ll = np.max(log_likes)
            if max_ll > -np.inf:
                log_posterior[i, j] = max_ll + np.log(
                    np.sum(np.exp(log_likes - max_ll))
                )

    # Normalize
    max_lp = np.max(log_posterior)
    posterior = np.exp(log_posterior - max_lp)
    posterior /= np.sum(posterior)

    marginal_ra = np.sum(posterior, axis=1)
    marginal_ra /= np.sum(marginal_ra)
    marginal_rb = np.sum(posterior, axis=0)
    marginal_rb /= np.sum(marginal_rb)

    result = {
        'posterior': posterior,
        'ra_grid': ra_grid,
        'rb_grid': rb_grid,
        'log_posterior': log_posterior,
        'marginal_ra': marginal_ra,
        'marginal_rb': marginal_rb,
        'd': d,
    }

    if marginalize_lam:
        result['lam_grid'] = lam_grid
    if marginalize_mu:
        result['mu_grid'] = mu_grid

    # Compute lam/mu marginals at MAP (r_a, r_b)
    if (marginalize_lam or marginalize_mu) and n_lam > 1 and n_mu > 1:
        i_map, j_map = np.unravel_index(np.argmax(posterior), posterior.shape)
        log_likes_map = np.zeros((n_lam, n_mu))
        for il in range(n_lam):
            for im in range(n_mu):
                log_likes_map[il, im] = compute_log_likelihood_multi(
                    counts, ra_grid[i_map], rb_grid[j_map],
                    lam_grid[il], mu_grid[im], d
                )
        max_ll = np.max(log_likes_map)
        likes_map = np.exp(np.clip(log_likes_map - max_ll, -700, 0))

        if marginalize_lam and n_lam > 1:
            marginal_lam = np.sum(likes_map, axis=1)
            s = np.sum(marginal_lam)
            result['marginal_lam'] = marginal_lam / s if s > 0 else marginal_lam

        if marginalize_mu and n_mu > 1:
            marginal_mu = np.sum(likes_map, axis=0)
            s = np.sum(marginal_mu)
            result['marginal_mu'] = marginal_mu / s if s > 0 else marginal_mu

    return result


def compute_accuracy_vs_d(
    r_a_true: float,
    r_b_true: float,
    lam: float,
    mu: float,
    N: int = 500,
    d_values: list = None,
    n_grid: int = 50,
    seed: int = 42
) -> dict:
    """
    Compute inference error for various values of d (Figure 9 analysis).

    For each d, performs inference on CTMC-simulated data and measures the
    Euclidean distance between the true rates and the posterior mode.

    Parameters
    ----------
    r_a_true, r_b_true : float
        True switching rates used to generate data.
    lam, mu : float
        Known fluorescence and background rates.
    N : int
        Number of data points.
    d_values : list of int
        Values of d to test (each must be power of 2).
    n_grid : int
        Grid resolution for inference.
    seed : int
        Random seed.

    Returns
    -------
    result : dict
        'd_values': list of d values tested
        'errors': corresponding Euclidean errors
        'posteriors': list of posterior dicts for each d
    """
    from .simulation import simulate_ctmc

    if d_values is None:
        d_values = [1, 2, 4, 8, 16]

    # Generate CTMC data
    counts, _, _ = simulate_ctmc(r_a_true, r_b_true, lam, mu, N, seed=seed)

    errors = []
    posteriors = []

    for d in d_values:
        result = compute_posterior_multi(
            counts, d=d,
            ra_range=(0.01, max(r_a_true * 3, 2.0)),
            rb_range=(0.01, max(r_b_true * 3, 2.0)),
            lam_known=lam,
            mu_known=mu,
            n_ra=n_grid,
            n_rb=n_grid,
            show_progress=False
        )

        # Find MAP estimate
        idx = np.unravel_index(np.argmax(result['posterior']), result['posterior'].shape)
        ra_map = result['ra_grid'][idx[0]]
        rb_map = result['rb_grid'][idx[1]]

        # Euclidean error
        error = np.sqrt((ra_map - r_a_true)**2 + (rb_map - r_b_true)**2)
        errors.append(error)
        posteriors.append(result)

    return {
        'd_values': d_values,
        'errors': errors,
        'posteriors': posteriors,
        'counts': counts
    }
