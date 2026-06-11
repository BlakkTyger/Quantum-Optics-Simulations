"""
DTMC Single-Step Bayesian Inference (Section 2 of the paper)
=============================================================

This module implements the Bayesian inference for the DTMC single-step model,
where the emitter state is fixed during each detector interval and switching
occurs only at interval boundaries.

Key Equations:
    - Posterior: P(alpha, beta | c, I_1) ∝ ∫∫ dλ dμ P(c | Ω₁)       [Eq. 8]
    - Likelihood via matrix product: P(c|Ω₁) = [1,1] ∏ R_t · D_0    [Eq. 16]
    - Transfer matrix R_t elements:                                    [Eq. 13]
        R_t[s_t, s_{t-1}] = P(c_t, s_t | s_{t-1}, Ω₁)

Mathematical Framework:
    The key insight is that summing over all possible state sequences (2^N terms)
    can be efficiently computed as a product of N 2×2 matrices. Each matrix R_t
    encodes the joint probability of observing count c_t AND transitioning to
    state s_t given the previous state s_{t-1}.

    R_t = [[P(c_t,s_t=0|s_{t-1}=0), P(c_t,s_t=0|s_{t-1}=1)],
           [P(c_t,s_t=1|s_{t-1}=0), P(c_t,s_t=1|s_{t-1}=1)]]

    where:
        P(c_t, s_t=0 | s_{t-1}=0) = (1-α) * Poisson(c_t | μ)
        P(c_t, s_t=1 | s_{t-1}=0) = α * Poisson(c_t | μ+λ)
        P(c_t, s_t=0 | s_{t-1}=1) = β * Poisson(c_t | μ)
        P(c_t, s_t=1 | s_{t-1}=1) = (1-β) * Poisson(c_t | μ+λ)

References:
    Geordy et al., New J. Phys. 21 (2019) 063001, Section 2
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional
from tqdm import tqdm
from .cache import disk_cache


def compute_log_poisson(counts: np.ndarray, rate: float) -> np.ndarray:
    """
    Compute log-Poisson probability for an array of counts at a given rate.

    Parameters
    ----------
    counts : np.ndarray
        Array of photon count values.
    rate : float
        Poisson rate parameter (must be > 0).

    Returns
    -------
    log_probs : np.ndarray
        Log of Poisson probability mass function evaluated at each count.
    """
    if rate <= 0:
        return np.full_like(counts, -np.inf, dtype=float)
    return poisson.logpmf(counts, rate)


def compute_transfer_matrices(
    counts: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
    mu: float
) -> np.ndarray:
    """
    Compute the transfer matrices R_t for all time steps.

    Each R_t is a 2x2 matrix encoding the joint probability of
    observing count c_t and transitioning to state s_t, given the
    previous state s_{t-1}.

    Parameters
    ----------
    counts : np.ndarray of shape (N,)
        Observed photon counts.
    alpha : float
        Switch-on probability (off→on).
    beta : float
        Switch-off probability (on→off).
    lam : float
        Fluorescence rate.
    mu : float
        Background rate.

    Returns
    -------
    R : np.ndarray of shape (N, 2, 2)
        Transfer matrices. R[t] is the 2x2 matrix for time step t.
        Convention: R[t][s_t, s_{t-1}] = P(c_t, s_t | s_{t-1}, Ω)

    Notes
    -----
    Matrix layout (rows = new state, columns = previous state):
        R_t = [[P(c_t,0|0), P(c_t,0|1)],   = [[(1-α)P_off, β·P_off    ],
               [P(c_t,1|0), P(c_t,1|1)]]      [ α·P_on,    (1-β)·P_on  ]]

    where P_off = Poisson(c_t|μ) and P_on = Poisson(c_t|μ+λ)
    """
    N = len(counts)

    # Poisson probabilities for off and on states
    log_p_off = compute_log_poisson(counts, mu)          # P(c_t | state=off)
    log_p_on = compute_log_poisson(counts, mu + lam)     # P(c_t | state=on)

    p_off = np.exp(log_p_off)
    p_on = np.exp(log_p_on)

    # Build transfer matrices
    # R[t, s_t, s_{t-1}] = P(c_t | s_{t-1}) * P(s_t | s_{t-1})
    # Poisson rate depends on s_{t-1} (column): off→Poisson(mu), on→Poisson(mu+lam)
    R = np.zeros((N, 2, 2))
    R[:, 0, 0] = (1 - alpha) * p_off   # s_{t-1}=off, stay off, counts~Poisson(mu)
    R[:, 0, 1] = beta * p_on           # s_{t-1}=on, switch off, counts~Poisson(mu+lam)
    R[:, 1, 0] = alpha * p_off         # s_{t-1}=off, switch on, counts~Poisson(mu)
    R[:, 1, 1] = (1 - beta) * p_on     # s_{t-1}=on, stay on, counts~Poisson(mu+lam)

    return R


def compute_log_likelihood_matrix_product(
    counts: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
    mu: float
) -> float:
    """
    Compute log P(c | Ω₁) using the matrix product formulation (Eq. 16).

    P(c | Ω₁) = [1,1] · R_N · R_{N-1} · ... · R_1 · D_0

    where D_0 is the initial state probability vector (assumed uniform = [0.5, 0.5]).

    Parameters
    ----------
    counts : np.ndarray of shape (N,)
        Observed photon counts.
    alpha : float
        Switch-on probability.
    beta : float
        Switch-off probability.
    lam : float
        Fluorescence rate.
    mu : float
        Background rate.

    Returns
    -------
    log_likelihood : float
        Log of the marginal likelihood P(c | Ω₁).

    Notes
    -----
    To avoid numerical underflow in the matrix products, we periodically
    rescale the running vector and accumulate the log of the scale factors.
    This is equivalent to computing log(P(c|Ω₁)) without overflow.
    """
    N = len(counts)

    # Initial state distribution (uniform prior on initial state)
    # D_0 = [P(s_0=0|Ω), P(s_0=1|Ω)]
    # Using stationary distribution: pi_off = beta/(alpha+beta), pi_on = alpha/(alpha+beta)
    if alpha + beta > 0:
        D0 = np.array([beta / (alpha + beta), alpha / (alpha + beta)])
    else:
        D0 = np.array([0.5, 0.5])

    # Compute transfer matrices
    R = compute_transfer_matrices(counts, alpha, beta, lam, mu)

    # Compute matrix product with periodic rescaling to avoid underflow
    # We compute: v = R_N · R_{N-1} · ... · R_1 · D_0
    # by iterating: v ← R_t · v (from t=1 to t=N)
    v = D0.copy()
    log_scale = 0.0

    for t in range(N):
        v = R[t] @ v
        # Rescale to prevent underflow
        scale = np.sum(v)
        if scale > 0:
            v /= scale
            log_scale += np.log(scale)
        else:
            return -np.inf

    # Final multiplication by [1, 1]
    result = np.sum(v)
    if result > 0:
        log_likelihood = log_scale + np.log(result)
    else:
        log_likelihood = -np.inf

    return log_likelihood

@disk_cache()
def compute_posterior_grid(
    counts: np.ndarray,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    beta_range: Tuple[float, float] = (0.0, 1.0),
    lam_range: Optional[Tuple[float, float]] = None,
    mu_range: Optional[Tuple[float, float]] = None,
    lam_known: Optional[float] = None,
    mu_known: Optional[float] = None,
    n_alpha: int = 100,
    n_beta: int = 100,
    n_lam: int = 50,
    n_mu: int = 50,
    show_progress: bool = True
) -> dict:
    """
    Compute the posterior distribution P(α₁, β₁ | c, I₁) on a grid.

    If λ and μ are known, computes P(α, β | c, λ, μ, I₁) directly.
    If unknown, marginalizes over them: P(α, β | c, I₁) = ∫∫ dλ dμ P(c|Ω₁)

    Performance: Uses a fully vectorized approach. For each (lam, mu) pair,
    ALL (alpha, beta) grid points are computed simultaneously by batching
    the matrix-vector forward pass over the entire (alpha, beta) grid.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon count time series.
    alpha_range : tuple
        Range (min, max) for alpha grid.
    beta_range : tuple
        Range (min, max) for beta grid.
    lam_range : tuple, optional
        Range for lambda marginalization. Required if lam_known is None.
    mu_range : tuple, optional
        Range for mu marginalization. Required if mu_known is None.
    lam_known : float, optional
        Known fluorescence rate (skips marginalization).
    mu_known : float, optional
        Known background rate (skips marginalization).
    n_alpha : int
        Number of grid points for alpha.
    n_beta : int
        Number of grid points for beta.
    n_lam : int
        Number of grid points for lambda (if marginalizing).
    n_mu : int
        Number of grid points for mu (if marginalizing).
    show_progress : bool
        Whether to show a progress bar.

    Returns
    -------
    result : dict
        'posterior' : 2D array of posterior probability (normalized)
        'alpha_grid' : 1D array of alpha values
        'beta_grid' : 1D array of beta values
        'log_posterior' : 2D array of log posterior (unnormalized)
        'marginal_alpha' : 1D marginal distribution of alpha
        'marginal_beta' : 1D marginal distribution of beta
        'marginal_lam' : 1D marginal of lambda (if marginalized)
        'marginal_mu' : 1D marginal of mu (if marginalized)
        'lam_grid' : lambda grid (if marginalized)
        'mu_grid' : mu grid (if marginalized)
    """
    # Create grids
    alpha_grid = np.linspace(alpha_range[0] + 1e-6, alpha_range[1] - 1e-6, n_alpha)
    beta_grid = np.linspace(beta_range[0] + 1e-6, beta_range[1] - 1e-6, n_beta)

    marginalize_lam = lam_known is None
    marginalize_mu = mu_known is None

    if marginalize_lam:
        if lam_range is None:
            raise ValueError("lam_range must be provided if lam_known is None")
        lam_grid = np.linspace(lam_range[0] + 1e-6, lam_range[1], n_lam)
    else:
        lam_grid = np.array([lam_known])
        n_lam = 1

    if marginalize_mu:
        if mu_range is None:
            raise ValueError("mu_range must be provided if mu_known is None")
        mu_grid = np.linspace(mu_range[0] + 1e-6, mu_range[1], n_mu)
    else:
        mu_grid = np.array([mu_known])
        n_mu = 1

    N = len(counts)

    # Precompute transition matrices T for all (alpha, beta) combinations.
    # T[i, j] is a 2x2 matrix: T[i,j][s_t, s_{t-1}] = P(s_t | s_{t-1})
    # Shape: (n_alpha, n_beta, 2, 2)
    T = np.zeros((n_alpha, n_beta, 2, 2))
    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            T[i, j, 0, 0] = 1 - alpha   # off → off
            T[i, j, 0, 1] = beta         # on → off
            T[i, j, 1, 0] = alpha        # off → on
            T[i, j, 1, 1] = 1 - beta    # on → on

    # Initial state vectors D0 for all (alpha, beta)
    # Shape: (n_alpha, n_beta, 2)
    D0 = np.zeros((n_alpha, n_beta, 2))
    for i, alpha in enumerate(alpha_grid):
        for j, beta in enumerate(beta_grid):
            total = alpha + beta
            if total > 0:
                D0[i, j, 0] = beta / total
                D0[i, j, 1] = alpha / total
            else:
                D0[i, j, :] = 0.5

    # Precompute Poisson PMFs for all unique counts and all rates
    unique_counts = np.unique(counts)
    count_to_idx = {int(c): i for i, c in enumerate(unique_counts)}
    count_indices = np.array([count_to_idx[int(c)] for c in counts])

    # p_off_table[im, c_idx] = Poisson(unique_counts[c_idx] | mu_grid[im])
    p_off_table = np.zeros((n_mu, len(unique_counts)))
    for im, mu_val in enumerate(mu_grid):
        p_off_table[im, :] = poisson.pmf(unique_counts, mu_val)

    # p_on_table[il, im, c_idx] = Poisson(unique_counts[c_idx] | mu_grid[im] + lam_grid[il])
    p_on_table = np.zeros((n_lam, n_mu, len(unique_counts)))
    for il, lam_val in enumerate(lam_grid):
        for im, mu_val in enumerate(mu_grid):
            p_on_table[il, im, :] = poisson.pmf(unique_counts, mu_val + lam_val)

    # Main computation: for each (lam, mu), compute log-likelihood for ALL (alpha, beta)
    # simultaneously using batched forward pass.
    # log_posterior_4d[i, j, il, im] = log P(c | alpha_i, beta_j, lam_il, mu_im)
    # We accumulate the marginalized result directly.
    log_posterior = np.full((n_alpha, n_beta), -np.inf)

    # Store log-likelihoods for marginal computation
    all_log_likes = np.full((n_alpha, n_beta, n_lam, n_mu), -np.inf) if (
        marginalize_lam or marginalize_mu) else None

    iterator = range(n_lam)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing posterior (DTMC single-step)",
                        total=n_lam)

    for il in iterator:
        for im in range(n_mu):
            # Poisson probabilities for this (lam, mu) at each time step
            # Shape: (N,) for off-state and on-state
            p_off_t = p_off_table[im, count_indices]   # P(c_t | off)
            p_on_t = p_on_table[il, im, count_indices]  # P(c_t | on)

            # Batched forward pass over all (alpha, beta) simultaneously
            # v shape: (n_alpha, n_beta, 2) — the running state vector
            v = D0.copy()
            log_scale = np.zeros((n_alpha, n_beta))

            for t in range(N):
                # Element-wise multiply v by Poisson probabilities:
                # v[:,:,0] *= p_off_t[t]  (column 0 = came from off state)
                # v[:,:,1] *= p_on_t[t]   (column 1 = came from on state)
                # Then multiply by transition matrix T
                # R_t @ v = T @ diag(p_off, p_on) @ v = T @ (v * [p_off, p_on])
                v[:, :, 0] *= p_off_t[t]
                v[:, :, 1] *= p_on_t[t]

                # Matrix multiply: v_new = T @ v (batched over alpha, beta)
                v_new = np.einsum('abij,abj->abi', T, v)
                v = v_new

                # Rescale to prevent underflow
                scale = v[:, :, 0] + v[:, :, 1]
                mask = scale > 0
                v[mask, 0] /= scale[mask]
                v[mask, 1] /= scale[mask]
                log_scale[mask] += np.log(scale[mask])
                # Where scale==0, log_scale stays (will give -inf effectively)
                log_scale[~mask] = -np.inf

            # Final: sum over states
            final_sum = v[:, :, 0] + v[:, :, 1]
            log_like = np.where(
                (final_sum > 0) & np.isfinite(log_scale),
                log_scale + np.log(np.maximum(final_sum, 1e-300)),
                -np.inf
            )

            # Store for marginalization
            if all_log_likes is not None:
                all_log_likes[:, :, il, im] = log_like

            # Accumulate into log_posterior using log-sum-exp over (lam, mu)
            # We'll do this at the end for all (lam, mu) together

    # If we have the full 4D array, marginalize over (lam, mu) using log-sum-exp
    if all_log_likes is not None:
        # Reshape for log-sum-exp over last two dims
        max_ll = np.max(all_log_likes, axis=(2, 3))
        # Avoid -inf issues
        valid = np.isfinite(max_ll)
        for i in range(n_alpha):
            for j in range(n_beta):
                if valid[i, j]:
                    shifted = all_log_likes[i, j, :, :] - max_ll[i, j]
                    log_posterior[i, j] = max_ll[i, j] + np.log(
                        np.sum(np.exp(shifted[np.isfinite(shifted)]))
                    )
    else:
        # Only one (lam, mu) point — log_like is the posterior directly
        log_posterior = log_like

    # Normalize the posterior
    max_lp = np.max(log_posterior[np.isfinite(log_posterior)])
    log_posterior_shifted = log_posterior - max_lp
    posterior = np.exp(np.clip(log_posterior_shifted, -700, 0))
    posterior /= np.sum(posterior)

    # Compute marginals
    marginal_alpha = np.sum(posterior, axis=1)
    marginal_alpha /= np.sum(marginal_alpha)
    marginal_beta = np.sum(posterior, axis=0)
    marginal_beta /= np.sum(marginal_beta)

    result = {
        'posterior': posterior,
        'alpha_grid': alpha_grid,
        'beta_grid': beta_grid,
        'log_posterior': log_posterior,
        'marginal_alpha': marginal_alpha,
        'marginal_beta': marginal_beta,
    }

    # Compute marginals for lambda and mu if they were marginalized
    if marginalize_lam or marginalize_mu:
        result['lam_grid'] = lam_grid
        result['mu_grid'] = mu_grid

        # Marginals of lam and mu at the MAP of (alpha, beta)
        i_map, j_map = np.unravel_index(np.argmax(posterior), posterior.shape)
        log_likes_map = all_log_likes[i_map, j_map, :, :]
        max_ll = np.max(log_likes_map[np.isfinite(log_likes_map)])
        likes_map = np.exp(np.clip(log_likes_map - max_ll, -700, 0))

        if marginalize_lam:
            marginal_lam = np.sum(likes_map, axis=1)
            s = np.sum(marginal_lam)
            result['marginal_lam'] = marginal_lam / s if s > 0 else marginal_lam

        if marginalize_mu:
            marginal_mu = np.sum(likes_map, axis=0)
            s = np.sum(marginal_mu)
            result['marginal_mu'] = marginal_mu / s if s > 0 else marginal_mu

    return result

@disk_cache()
def compute_posterior_known_rates(
    counts: np.ndarray,
    lam: float,
    mu: float,
    n_alpha: int = 200,
    n_beta: int = 200,
    alpha_range: Tuple[float, float] = (0.0, 1.0),
    beta_range: Tuple[float, float] = (0.0, 1.0),
    show_progress: bool = True
) -> dict:
    """
    Compute posterior P(α, β | c, λ, μ, I₁) when λ and μ are known.

    This is a simplified version that avoids marginalization over rates,
    useful for faster computation and for the state inference algorithm.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    lam : float
        Known fluorescence rate.
    mu : float
        Known background rate.
    n_alpha : int
        Grid resolution for alpha.
    n_beta : int
        Grid resolution for beta.
    alpha_range : tuple
        Range for alpha.
    beta_range : tuple
        Range for beta.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    result : dict
        Same structure as compute_posterior_grid but without lam/mu marginals.
    """
    return compute_posterior_grid(
        counts,
        alpha_range=alpha_range,
        beta_range=beta_range,
        lam_known=lam,
        mu_known=mu,
        n_alpha=n_alpha,
        n_beta=n_beta,
        show_progress=show_progress
    )
