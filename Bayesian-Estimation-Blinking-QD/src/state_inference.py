"""
State Inference (Section 5 of the paper)
==========================================

This module determines the underlying hidden state at each time step given
all observed data. Unlike threshold analysis which gives a binary assignment,
Bayesian state inference provides the probability distribution over states.

Key Equation (Eq. 57):
    P(s_k = a | c, I₁) = (1/Z) ∫ dα₁ dβ₁ dλ dμ
        [1,1] · R_N · ... · R_{k+1} · R_k^{(a)} · R_{k-1} · ... · R_1 · D_0

    where R_k^{(a)} is the transfer matrix R_k with only the row corresponding
    to state a retained (other row zeroed out):

        R_k^{(0)} = [[R_{k,00}, R_{k,01}],   (only off-state row kept)
                     [0,        0       ]]

        R_k^{(1)} = [[0,        0       ],   (only on-state row kept)
                     [R_{k,10}, R_{k,11}]]

Physical Interpretation:
    This computes the posterior probability that the emitter was in state 'a'
    at time step k, using ALL the observed data (past and future). This is
    a smoothing operation (as opposed to filtering which uses only past data).
    The result automatically accounts for uncertainty in the model parameters
    by marginalizing over them.

Algorithm:
    1. Compute forward vectors: v_forward[k] = R_k · ... · R_1 · D_0
    2. Compute backward vectors: v_backward[k] = [1,1] · R_N · ... · R_{k+1}
    3. State probability: P(s_k=a) ∝ v_backward[k] · R_k^{(a)} · v_forward[k-1]

    This forward-backward algorithm is equivalent to Eq. 57 but more efficient
    as it avoids redundant computation.

References:
    Geordy et al., New J. Phys. 21 (2019) 063001, Section 5
"""

import numpy as np
from scipy.stats import poisson
from typing import Tuple, Optional
from tqdm import tqdm
from .cache import disk_cache

from .dtmc_single import compute_transfer_matrices


def forward_backward_single_step(
    counts: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
    mu: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the forward-backward algorithm for the DTMC single-step model.

    This efficiently computes all the quantities needed for state inference
    without redundant matrix multiplications.

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
    state_probs : np.ndarray of shape (N, 2)
        state_probs[k, a] = P(s_k = a | c, α, β, λ, μ)
        Normalized so that state_probs[k, 0] + state_probs[k, 1] = 1.
    forward_vectors : np.ndarray of shape (N+1, 2)
        Forward messages (scaled).
    backward_vectors : np.ndarray of shape (N+1, 2)
        Backward messages (scaled).

    Notes
    -----
    Forward pass: Computes α_k = R_k · α_{k-1} (with rescaling)
    Backward pass: Computes β_k = R_{k+1}^T · β_{k+1} (with rescaling)
    State probability: P(s_k=a) ∝ Σ_{s_{k-1}} R_k[a, s_{k-1}] · α_{k-1}[s_{k-1}]
                                   × β_k[a]
    """
    N = len(counts)

    # Compute all transfer matrices
    R = compute_transfer_matrices(counts, alpha, beta, lam, mu)

    # Initial state distribution (stationary)
    if alpha + beta > 0:
        D0 = np.array([beta / (alpha + beta), alpha / (alpha + beta)])
    else:
        D0 = np.array([0.5, 0.5])

    # Forward pass: forward[k] = scaled version of R_k · ... · R_1 · D_0
    forward = np.zeros((N + 1, 2))
    forward_scale = np.zeros(N + 1)
    forward[0] = D0
    forward_scale[0] = 1.0

    for k in range(N):
        forward[k + 1] = R[k] @ forward[k]
        s = np.sum(forward[k + 1])
        if s > 0:
            forward[k + 1] /= s
            forward_scale[k + 1] = s
        else:
            forward_scale[k + 1] = 1e-300

    # Backward pass: backward[k] represents [1,1] · R_N · ... · R_{k+1}
    # We work with row vectors: backward[k] = backward[k+1] · R_{k+1}
    # Or equivalently, backward[k]^T = R_{k+1}^T · backward[k+1]^T
    backward = np.zeros((N + 1, 2))
    backward_scale = np.zeros(N + 1)
    backward[N] = np.array([1.0, 1.0])  # Final: [1, 1]
    backward_scale[N] = 1.0

    for k in range(N - 1, -1, -1):
        # backward[k] = backward[k+1] @ R[k+1] if we think of it as row vector
        # But we need: backward[k] such that backward[k] · (R_{k+1} · v) gives
        # the marginal. Actually:
        # backward[k][a] = Σ_b backward[k+1][b] · R[k+1][b, a] (if k+1 < N)
        # For the last step: backward[N][a] = 1
        if k < N - 1:
            backward[k] = R[k + 1].T @ backward[k + 1]
        else:
            backward[k] = np.array([1.0, 1.0])

        s = np.sum(backward[k])
        if s > 0:
            backward[k] /= s
            backward_scale[k] = s
        else:
            backward_scale[k] = 1e-300

    # Compute state probabilities using forward-backward
    # P(s_k = a | c) ∝ [R_k[a, :] · forward[k-1]] · backward[k][a]
    # where forward[k-1] is the forward vector BEFORE applying R_k
    state_probs = np.zeros((N, 2))

    for k in range(N):
        for a in range(2):
            # Contribution from R_k with only row 'a' active
            # P(s_k=a, c_k | s_{k-1}) summed over s_{k-1}
            contribution = R[k][a, :] @ forward[k]
            state_probs[k, a] = contribution * backward[k][a]

        # Normalize
        total = np.sum(state_probs[k])
        if total > 0:
            state_probs[k] /= total
        else:
            state_probs[k] = 0.5

    return state_probs, forward, backward

@disk_cache()
def infer_states_marginalized(
    counts: np.ndarray,
    alpha_range: Tuple[float, float] = (0.001, 0.5),
    beta_range: Tuple[float, float] = (0.001, 0.5),
    lam_known: Optional[float] = None,
    mu_known: Optional[float] = None,
    lam_range: Optional[Tuple[float, float]] = None,
    mu_range: Optional[Tuple[float, float]] = None,
    n_alpha: int = 30,
    n_beta: int = 30,
    n_lam: int = 20,
    n_mu: int = 20,
    show_progress: bool = True
) -> np.ndarray:
    """
    Infer hidden states by marginalizing over model parameters (Eq. 57).

    P(s_k = a | c, I₁) = (1/Z) ∫ dα dβ dλ dμ P(s_k=a, c | Ω₁)

    This performs full Bayesian inference on the states, integrating out
    all unknown parameters to give the most robust state estimates.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    alpha_range : tuple
        Range for alpha grid.
    beta_range : tuple
        Range for beta grid.
    lam_known : float, optional
        Known fluorescence rate.
    mu_known : float, optional
        Known background rate.
    lam_range : tuple, optional
        Range for lambda (if unknown).
    mu_range : tuple, optional
        Range for mu (if unknown).
    n_alpha, n_beta : int
        Grid resolution for switching probabilities.
    n_lam, n_mu : int
        Grid resolution for rates.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    state_probs : np.ndarray of shape (N, 2)
        state_probs[k, a] = P(s_k = a | c, I₁)
        Marginalized over all model parameters.
    """
    N = len(counts)

    alpha_grid = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    beta_grid = np.linspace(beta_range[0], beta_range[1], n_beta)

    if lam_known is not None:
        lam_grid = np.array([lam_known])
    else:
        if lam_range is None:
            raise ValueError("lam_range required when lam_known is None")
        lam_grid = np.linspace(lam_range[0] + 0.1, lam_range[1], n_lam)

    if mu_known is not None:
        mu_grid = np.array([mu_known])
    else:
        if mu_range is None:
            raise ValueError("mu_range required when mu_known is None")
        mu_grid = np.linspace(mu_range[0] + 0.1, mu_range[1], n_mu)

    # Accumulate weighted state probabilities over the parameter grid
    # P(s_k=a|c) ∝ Σ_{params} P(s_k=a, c | params) · P(params)
    # With uniform priors, this becomes Σ P(s_k=a | c, params) · P(c | params)
    accumulated_state_probs = np.zeros((N, 2))
    total_weight = 0.0

    # Iterate over parameter grid
    n_total = len(alpha_grid) * len(beta_grid) * len(lam_grid) * len(mu_grid)
    iterator = enumerate(
        (a, b, l, m)
        for a in alpha_grid
        for b in beta_grid
        for l in lam_grid
        for m in mu_grid
    )

    if show_progress:
        from tqdm import tqdm as tqdm_bar
        pbar = tqdm_bar(total=n_total, desc="State inference (marginalized)")

    for idx, (alpha, beta, lam, mu) in iterator:
        # Compute state probabilities for this parameter setting
        state_probs, _, _ = forward_backward_single_step(counts, alpha, beta, lam, mu)

        # Compute likelihood P(c | params) as weight
        # Use the forward pass final value as log-likelihood proxy
        from .dtmc_single import compute_log_likelihood_matrix_product
        log_like = compute_log_likelihood_matrix_product(counts, alpha, beta, lam, mu)

        if log_like > -np.inf:
            # Use likelihood as weight (in log space for stability)
            weight = np.exp(log_like - (-100))  # shift for numerical stability
            accumulated_state_probs += weight * state_probs
            total_weight += weight

        if show_progress:
            pbar.update(1)

    if show_progress:
        pbar.close()

    # Normalize
    if total_weight > 0:
        accumulated_state_probs /= total_weight

    # Final normalization per time step
    for k in range(N):
        total = np.sum(accumulated_state_probs[k])
        if total > 0:
            accumulated_state_probs[k] /= total
        else:
            accumulated_state_probs[k] = 0.5

    return accumulated_state_probs

@disk_cache()
def infer_states_known_params(
    counts: np.ndarray,
    alpha: float,
    beta: float,
    lam: float,
    mu: float
) -> np.ndarray:
    """
    Infer hidden states with known model parameters.

    A simplified version that skips marginalization—useful when parameters
    are known (e.g., from prior inference or simulation ground truth).

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    alpha, beta : float
        Known switching probabilities.
    lam, mu : float
        Known fluorescence and background rates.

    Returns
    -------
    state_probs : np.ndarray of shape (N, 2)
        state_probs[k, a] = P(s_k = a | c, α, β, λ, μ)
    """
    state_probs, _, _ = forward_backward_single_step(counts, alpha, beta, lam, mu)
    return state_probs


def infer_states_with_convergence(
    counts: np.ndarray,
    lam: float,
    mu: float,
    n_alpha: int = 50,
    n_beta: int = 50,
    alpha_range: Tuple[float, float] = (0.001, 0.5),
    beta_range: Tuple[float, float] = (0.001, 0.5),
    data_fractions: list = None,
    show_progress: bool = True
) -> dict:
    """
    Demonstrate convergence of state inference with increasing data (Figure 11).

    Runs the state inference using progressively more data points, showing
    how the posterior on both the states and switching probabilities converges.

    Parameters
    ----------
    counts : np.ndarray
        Full observed photon count sequence.
    lam : float
        Known fluorescence rate.
    mu : float
        Known background rate.
    n_alpha, n_beta : int
        Grid resolution.
    alpha_range, beta_range : tuple
        Ranges for switching probabilities.
    data_fractions : list of float
        Fractions of data to use (e.g., [0.25, 0.5, 0.75, 1.0]).
    show_progress : bool
        Show progress bars.

    Returns
    -------
    result : dict
        'fractions': data fractions used
        'state_probs': list of state probability arrays for each fraction
        'posteriors': list of posterior dicts for each fraction
    """
    from .dtmc_single import compute_posterior_known_rates

    if data_fractions is None:
        data_fractions = [0.25, 0.5, 0.75, 1.0]

    N = len(counts)
    results_list = []

    for frac in data_fractions:
        n_use = max(10, int(N * frac))
        counts_subset = counts[:n_use]

        # Compute posterior on alpha, beta
        posterior = compute_posterior_known_rates(
            counts_subset, lam, mu,
            n_alpha=n_alpha, n_beta=n_beta,
            alpha_range=alpha_range, beta_range=beta_range,
            show_progress=show_progress
        )

        # Find MAP parameters
        idx = np.unravel_index(
            np.argmax(posterior['posterior']),
            posterior['posterior'].shape
        )
        alpha_map = posterior['alpha_grid'][idx[0]]
        beta_map = posterior['beta_grid'][idx[1]]

        # Infer states using MAP parameters on the full data up to this point
        state_probs = infer_states_known_params(
            counts_subset, alpha_map, beta_map, lam, mu
        )

        results_list.append({
            'n_data': n_use,
            'state_probs': state_probs,
            'posterior': posterior,
            'alpha_map': alpha_map,
            'beta_map': beta_map,
        })

    return {
        'fractions': data_fractions,
        'results': results_list,
    }
