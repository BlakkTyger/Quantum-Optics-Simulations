"""
CTMC Bayesian Inference (Section 3 of the paper)
=================================================

This module implements the Bayesian inference for the Continuous Time Markov
Chain (CTMC) model, where the emitter can switch states at any continuous
time point within a detector interval.

Key Physics:
    The CTMC is governed by the rate matrix Q:
        Q = [[-r_a,  r_a],
             [ r_b, -r_b]]

    where r_a is the switch-on rate and r_b is the switch-off rate.
    The formal solution P(t) = exp(Qt) gives transition probabilities.

Key Equations:
    - R_ab(f): Distribution of fraction f spent in the on state given boundary
      states a and b. Derived by summing over all possible numbers of switches.

    - R_00(f) = δ(f)·e^{-r_a} + e^{-r_a(1-f)-r_b·f} · √((f-1)r_a·r_b/f)
                · J_1(2√((f-1)f·r_a·r_b))                              [Eq. 38]

    - R_01(f) = r_a · J_0(2√((f-1)f·r_a·r_b)) · e^{-r_a(1-f)-r_b·f}  [Eq. 39]

    - R_10(f) = r_b · J_0(2√((f-1)f·r_a·r_b)) · e^{-r_a(1-f)-r_b·f}  [Eq. 40]

    - R_11(f) = δ(1-f)·e^{-r_b} + e^{-r_a(1-f)-r_b·f} · √(f·r_a·r_b/(f-1))
                · J_1(2√((f-1)f·r_a·r_b))                              [Eq. 41]

    - Final element: P(c_n, s_n | s_{n-1}) = ∫₀¹ df Poisson(c_n|μ+f·λ)·R_{s_{n-1},s_n}(f)
                                                                         [Eq. 42]

    Note: The argument to the Bessel functions involves √((f-1)f) which equals
    √(f(1-f)) since f∈[0,1], making the argument real and non-negative.
    The paper uses the convention (f-1) which gives negative values inside the
    square root, but mathematically √((f-1)f·r_a·r_b) = i·√(f(1-f)·r_a·r_b)
    and J_n with imaginary argument relates to modified Bessel functions I_n.
    
    Actually, re-reading carefully: the paper writes √((f-1)·f·r_a·r_b) where
    f∈[0,1], so (f-1) ≤ 0, making the product (f-1)·f ≤ 0. Taking the square
    root of a negative number gives imaginary values. However, J_1(ix) = i·I_1(x)
    and the overall expression remains real. We implement using modified Bessel
    functions I_0 and I_1 for numerical stability.

References:
    Geordy et al., New J. Phys. 21 (2019) 063001, Section 3
    Jaynes E T 2003, Probability Theory, Ch. 18 (Laplace transform derivation)
"""

import numpy as np
from scipy.special import iv as bessel_i  # Modified Bessel function I_v(x)
from scipy.stats import poisson
from typing import Tuple, Optional
from tqdm import tqdm
from .cache import disk_cache


def R00(f: np.ndarray, r_a: float, r_b: float) -> np.ndarray:
    """
    Compute R_00(f): probability density for fraction f on, given start=off, end=off.

    R_00(f) = δ(f)·e^{-r_a} + e^{-r_a(1-f)-r_b·f} · √(r_a·r_b·(1-f)/f)
              · I_1(2√(f(1-f)·r_a·r_b))

    The delta function contribution (no switches, entire interval off) is handled
    separately during integration. This function returns only the continuous part.

    Parameters
    ----------
    f : np.ndarray
        Fraction of interval spent in on state, f ∈ (0, 1).
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.

    Returns
    -------
    R : np.ndarray
        Probability density at each f value (continuous part only).

    Notes
    -----
    For the discrete (delta function) part at f=0:
        weight = e^{-r_a}  (probability of no switch during entire interval)
    
    The Bessel function identity: J_1(ix) = i·I_1(x), so:
        √((f-1)f·r_a·r_b)/f · J_1(2√((f-1)f·r_a·r_b))
    becomes (using z = 2√(f(1-f)·r_a·r_b)):
        √(r_a·r_b·(1-f)/f) · I_1(z)
    """
    # Avoid division by zero at boundaries
    f = np.clip(f, 1e-15, 1 - 1e-15)

    z = 2.0 * np.sqrt(f * (1 - f) * r_a * r_b)
    exponential = np.exp(-r_a * (1 - f) - r_b * f)

    # Prefactor: sqrt(r_a * r_b * (1-f) / f)
    prefactor = np.sqrt(r_a * r_b * (1 - f) / f)

    return exponential * prefactor * bessel_i(1, z)


def R01(f: np.ndarray, r_a: float, r_b: float) -> np.ndarray:
    """
    Compute R_01(f): probability density for fraction f on, given start=off, end=on.

    R_01(f) = r_a · I_0(2√(f(1-f)·r_a·r_b)) · e^{-r_a(1-f)-r_b·f}

    Parameters
    ----------
    f : np.ndarray
        Fraction of interval spent in on state, f ∈ (0, 1).
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.

    Returns
    -------
    R : np.ndarray
        Probability density at each f value.

    Notes
    -----
    J_0(ix) = I_0(x), so the conversion is straightforward.
    This term has no delta function contribution.
    """
    f = np.clip(f, 1e-15, 1 - 1e-15)
    z = 2.0 * np.sqrt(f * (1 - f) * r_a * r_b)
    exponential = np.exp(-r_a * (1 - f) - r_b * f)
    return r_a * bessel_i(0, z) * exponential


def R10(f: np.ndarray, r_a: float, r_b: float) -> np.ndarray:
    """
    Compute R_10(f): probability density for fraction f on, given start=on, end=off.

    R_10(f) = r_b · I_0(2√(f(1-f)·r_a·r_b)) · e^{-r_a(1-f)-r_b·f}

    Parameters
    ----------
    f : np.ndarray
        Fraction of interval spent in on state, f ∈ (0, 1).
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.

    Returns
    -------
    R : np.ndarray
        Probability density at each f value.
    """
    f = np.clip(f, 1e-15, 1 - 1e-15)
    z = 2.0 * np.sqrt(f * (1 - f) * r_a * r_b)
    exponential = np.exp(-r_a * (1 - f) - r_b * f)
    return r_b * bessel_i(0, z) * exponential


def R11(f: np.ndarray, r_a: float, r_b: float) -> np.ndarray:
    """
    Compute R_11(f): probability density for fraction f on, given start=on, end=on.

    R_11(f) = δ(1-f)·e^{-r_b} + e^{-r_a(1-f)-r_b·f} · √(r_a·r_b·f/(1-f))
              · I_1(2√(f(1-f)·r_a·r_b))

    The delta function contribution (no switches, entire interval on) is handled
    separately. This returns only the continuous part.

    Parameters
    ----------
    f : np.ndarray
        Fraction of interval spent in on state, f ∈ (0, 1).
    r_a : float
        Switch-on rate.
    r_b : float
        Switch-off rate.

    Returns
    -------
    R : np.ndarray
        Probability density at each f (continuous part only).

    Notes
    -----
    For the discrete (delta function) part at f=1:
        weight = e^{-r_b}  (probability of no switch during entire interval)
    """
    f = np.clip(f, 1e-15, 1 - 1e-15)
    z = 2.0 * np.sqrt(f * (1 - f) * r_a * r_b)
    exponential = np.exp(-r_a * (1 - f) - r_b * f)

    # Prefactor: sqrt(r_a * r_b * f / (1-f))
    prefactor = np.sqrt(r_a * r_b * f / (1 - f))

    return exponential * prefactor * bessel_i(1, z)


def precompute_quadrature(n_quad: int = 80):
    """
    Precompute Gauss-Legendre quadrature nodes and weights on [0, 1].

    Parameters
    ----------
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    nodes : np.ndarray of shape (n_quad,)
        Quadrature nodes in [0, 1].
    weights : np.ndarray of shape (n_quad,)
        Quadrature weights.
    """
    from numpy.polynomial.legendre import leggauss
    # Gauss-Legendre on [-1, 1]
    x, w = leggauss(n_quad)
    # Transform to [0, 1]
    nodes = (x + 1) / 2
    weights = w / 2
    return nodes, weights


def compute_transfer_matrix_vectorized(
    c_n: int,
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    quad_nodes: np.ndarray,
    quad_weights: np.ndarray
) -> np.ndarray:
    """
    Compute the full 2x2 transfer matrix R_n using vectorized quadrature.

    R_n[s_n, s_{n-1}] = P(c_n, s_n | s_{n-1}, Ω_c)
                      = ∫₀¹ df · Poisson(c_n | μ + f·λ) · R_{s_{n-1},s_n}(f)

    Uses precomputed Gauss-Legendre nodes/weights for fast evaluation.

    Parameters
    ----------
    c_n : int
        Observed count for this interval.
    r_a, r_b : float
        Switching rates.
    lam, mu : float
        Fluorescence and background rates.
    quad_nodes : np.ndarray
        Quadrature nodes in [0, 1].
    quad_weights : np.ndarray
        Quadrature weights.

    Returns
    -------
    R_n : np.ndarray of shape (2, 2)
        Transfer matrix with convention R_n[new_state, prev_state].
    """
    f = quad_nodes  # shape (n_quad,)

    # Poisson probabilities at all quadrature points
    poisson_vals = poisson.pmf(c_n, mu + f * lam)  # shape (n_quad,)

    # R_ab(f) for all (a,b) combinations at all quadrature points
    r00_vals = R00(f, r_a, r_b)  # shape (n_quad,)
    r01_vals = R01(f, r_a, r_b)
    r10_vals = R10(f, r_a, r_b)
    r11_vals = R11(f, r_a, r_b)

    # Integrate: sum(weights * Poisson * R_ab)
    pw = poisson_vals * quad_weights  # shape (n_quad,)

    R_n = np.zeros((2, 2))

    # Continuous contributions
    R_n[0, 0] = np.dot(pw, r00_vals)
    R_n[1, 0] = np.dot(pw, r01_vals)
    R_n[0, 1] = np.dot(pw, r10_vals)
    R_n[1, 1] = np.dot(pw, r11_vals)

    # Delta function contributions
    # R_00: delta(f) * e^{-r_a} → f=0, Poisson(c_n | mu)
    R_n[0, 0] += np.exp(-r_a) * poisson.pmf(c_n, mu)
    # R_11: delta(1-f) * e^{-r_b} → f=1, Poisson(c_n | mu+lam)
    R_n[1, 1] += np.exp(-r_b) * poisson.pmf(c_n, mu + lam)

    return R_n


def compute_log_likelihood_ctmc(
    counts: np.ndarray,
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    n_quad: int = 80
) -> float:
    """
    Compute log P(c | Ω_c) for the CTMC model using matrix products (Eq. 29).

    P(c | Ω_c) = [1,1] · ∏_{n=1}^{N} R_n · D_0

    Uses vectorized Gauss-Legendre quadrature for the integrals.

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
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    log_likelihood : float
        Log marginal likelihood.
    """
    N = len(counts)
    quad_nodes, quad_weights = precompute_quadrature(n_quad)

    # Initial state distribution (stationary)
    total_rate = r_a + r_b
    if total_rate > 0:
        D0 = np.array([r_b / total_rate, r_a / total_rate])
    else:
        D0 = np.array([0.5, 0.5])

    # Precompute transfer matrices for unique count values
    unique_counts = np.unique(counts)
    R_cache = {}
    for c in unique_counts:
        R_cache[int(c)] = compute_transfer_matrix_vectorized(
            int(c), r_a, r_b, lam, mu, quad_nodes, quad_weights
        )

    # Iterative matrix-vector product with rescaling
    v = D0.copy()
    log_scale = 0.0

    for n in range(N):
        R_n = R_cache[int(counts[n])]
        v = R_n @ v

        # Rescale
        scale = v[0] + v[1]
        if scale > 0:
            v /= scale
            log_scale += np.log(scale)
        else:
            return -np.inf

    result = v[0] + v[1]
    if result > 0:
        return log_scale + np.log(result)
    else:
        return -np.inf

@disk_cache()
def compute_posterior_ctmc(
    counts: np.ndarray,
    ra_range: Tuple[float, float] = (0.01, 8.0),
    rb_range: Tuple[float, float] = (0.01, 8.0),
    lam_range: Optional[Tuple[float, float]] = None,
    mu_range: Optional[Tuple[float, float]] = None,
    lam_known: Optional[float] = None,
    mu_known: Optional[float] = None,
    n_ra: int = 70,
    n_rb: int = 70,
    n_lam: int = 70,
    n_mu: int = 70,
    show_progress: bool = True
) -> dict:
    """
    Compute the posterior P(r_a, r_b | c, I_c) for the CTMC model.

    Parameters
    ----------
    counts : np.ndarray
        Observed photon counts.
    ra_range : tuple
        Range for r_a.
    rb_range : tuple
        Range for r_b.
    lam_range : tuple, optional
        Range for lambda (if marginalizing).
    mu_range : tuple, optional
        Range for mu (if marginalizing).
    lam_known : float, optional
        Known fluorescence rate.
    mu_known : float, optional
        Known background rate.
    n_ra, n_rb : int
        Grid resolution for rates.
    n_lam, n_mu : int
        Grid resolution for fluorescence/background rates.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    result : dict
        'posterior' : 2D normalized posterior
        'ra_grid', 'rb_grid' : grid values
        'marginal_ra', 'marginal_rb' : 1D marginals
        And optionally 'marginal_lam', 'marginal_mu'
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

    # Precompute quadrature
    n_quad = 60
    quad_nodes, quad_weights = precompute_quadrature(n_quad)

    # Precompute unique counts and their indices
    unique_counts = np.unique(counts)
    count_to_idx = {int(c): i for i, c in enumerate(unique_counts)}
    count_indices = np.array([count_to_idx[int(c)] for c in counts])
    n_unique = len(unique_counts)
    N = len(counts)

    # Precompute Poisson values for all unique counts at all quadrature nodes
    # for all (lam, mu) combinations.
    # Shape: (n_lam, n_mu, n_unique, n_quad) — Poisson(c | mu + f * lam)
    # Plus edge values: Poisson(c | mu) and Poisson(c | mu + lam)
    poisson_quad = np.zeros((n_lam, n_mu, n_unique, n_quad))
    poisson_f0 = np.zeros((n_lam, n_mu, n_unique))  # f=0: Poisson(c | mu)
    poisson_f1 = np.zeros((n_lam, n_mu, n_unique))  # f=1: Poisson(c | mu+lam)

    for il, lam_val in enumerate(lam_grid):
        for im, mu_val in enumerate(mu_grid):
            for ic, c_val in enumerate(unique_counts):
                rates = mu_val + quad_nodes * lam_val
                poisson_quad[il, im, ic, :] = poisson.pmf(int(c_val), rates)
                poisson_f0[il, im, ic] = poisson.pmf(int(c_val), mu_val)
                poisson_f1[il, im, ic] = poisson.pmf(int(c_val), mu_val + lam_val)

    iterator = range(n_ra)
    if show_progress:
        iterator = tqdm(iterator, desc="Computing posterior (CTMC)")

    for i in iterator:
        r_a = ra_grid[i]
        for j in range(n_rb):
            r_b = rb_grid[j]

            # Precompute R_ab(f) at quadrature nodes for this (r_a, r_b)
            r00_vals = R00(quad_nodes, r_a, r_b)  # shape (n_quad,)
            r01_vals = R01(quad_nodes, r_a, r_b)
            r10_vals = R10(quad_nodes, r_a, r_b)
            r11_vals = R11(quad_nodes, r_a, r_b)

            # Delta contributions
            delta_00 = np.exp(-r_a)  # for f=0 (off→off)
            delta_11 = np.exp(-r_b)  # for f=1 (on→on)

            # Compute transfer matrices for ALL (lam, mu) and all unique counts
            # R_matrices[il, im, ic, s_t, s_{t-1}]
            # Using einsum-style: integral = sum_q (poisson_q * weight_q * R_ab_q)
            # Weighted R values: shape (n_quad,) broadcast with poisson_quad shape (n_lam, n_mu, n_unique, n_quad)
            wR00 = quad_weights * r00_vals  # shape (n_quad,)
            wR01 = quad_weights * r01_vals
            wR10 = quad_weights * r10_vals
            wR11 = quad_weights * r11_vals

            # Transfer matrices: shape (n_lam, n_mu, n_unique, 2, 2)
            # Element [il, im, ic, s_t, s_{t-1}] = integral + delta
            R_all = np.zeros((n_lam, n_mu, n_unique, 2, 2))

            # Continuous integrals: sum over quadrature points
            R_all[:, :, :, 0, 0] = np.einsum('lmcq,q->lmc', poisson_quad, wR00)
            R_all[:, :, :, 1, 0] = np.einsum('lmcq,q->lmc', poisson_quad, wR01)
            R_all[:, :, :, 0, 1] = np.einsum('lmcq,q->lmc', poisson_quad, wR10)
            R_all[:, :, :, 1, 1] = np.einsum('lmcq,q->lmc', poisson_quad, wR11)

            # Add delta contributions
            R_all[:, :, :, 0, 0] += delta_00 * poisson_f0  # f=0 → Poisson(c|mu)
            R_all[:, :, :, 1, 1] += delta_11 * poisson_f1  # f=1 → Poisson(c|mu+lam)

            # Now run forward pass batched over (lam, mu)
            # v shape: (n_lam, n_mu, 2)
            total_rate = r_a + r_b
            if total_rate > 0:
                D0 = np.array([r_b / total_rate, r_a / total_rate])
            else:
                D0 = np.array([0.5, 0.5])

            v = np.broadcast_to(D0, (n_lam, n_mu, 2)).copy()
            log_scale = np.zeros((n_lam, n_mu))

            for t in range(N):
                ic = count_indices[t]
                # R_t for this time step: shape (n_lam, n_mu, 2, 2)
                R_t = R_all[:, :, ic, :, :]

                # Batched matrix-vector: v_new = R_t @ v
                v_new = np.einsum('lmij,lmj->lmi', R_t, v)
                v = v_new

                # Rescale
                scale = v[:, :, 0] + v[:, :, 1]
                mask = scale > 0
                v[mask, 0] /= scale[mask]
                v[mask, 1] /= scale[mask]
                log_scale[mask] += np.log(scale[mask])
                log_scale[~mask] = -np.inf

            # Final sum
            final_sum = v[:, :, 0] + v[:, :, 1]
            log_likes = np.where(
                (final_sum > 0) & np.isfinite(log_scale),
                log_scale + np.log(np.maximum(final_sum, 1e-300)),
                -np.inf
            )

            # Marginalize over (lam, mu)
            max_ll = np.max(log_likes)
            if max_ll > -np.inf:
                log_posterior[i, j] = max_ll + np.log(
                    np.sum(np.exp(log_likes - max_ll))
                )

    # Normalize
    max_lp = np.max(log_posterior)
    posterior = np.exp(log_posterior - max_lp)
    posterior /= np.sum(posterior)

    # Marginals
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
    }

    if marginalize_lam:
        result['lam_grid'] = lam_grid
    if marginalize_mu:
        result['mu_grid'] = mu_grid

    # Compute lam/mu marginals at the MAP (r_a, r_b) point
    if (marginalize_lam or marginalize_mu) and n_lam > 1 and n_mu > 1:
        i_map, j_map = np.unravel_index(np.argmax(posterior), posterior.shape)
        r_a_map = ra_grid[i_map]
        r_b_map = rb_grid[j_map]

        # Re-run forward pass at MAP (r_a, r_b) to get log_likes(lam, mu)
        r00_vals = R00(quad_nodes, r_a_map, r_b_map)
        r01_vals = R01(quad_nodes, r_a_map, r_b_map)
        r10_vals = R10(quad_nodes, r_a_map, r_b_map)
        r11_vals = R11(quad_nodes, r_a_map, r_b_map)
        delta_00 = np.exp(-r_a_map)
        delta_11 = np.exp(-r_b_map)

        wR00 = quad_weights * r00_vals
        wR01 = quad_weights * r01_vals
        wR10 = quad_weights * r10_vals
        wR11 = quad_weights * r11_vals

        R_all_map = np.zeros((n_lam, n_mu, n_unique, 2, 2))
        R_all_map[:, :, :, 0, 0] = np.einsum('lmcq,q->lmc', poisson_quad, wR00)
        R_all_map[:, :, :, 1, 0] = np.einsum('lmcq,q->lmc', poisson_quad, wR01)
        R_all_map[:, :, :, 0, 1] = np.einsum('lmcq,q->lmc', poisson_quad, wR10)
        R_all_map[:, :, :, 1, 1] = np.einsum('lmcq,q->lmc', poisson_quad, wR11)
        R_all_map[:, :, :, 0, 0] += delta_00 * poisson_f0
        R_all_map[:, :, :, 1, 1] += delta_11 * poisson_f1

        total_rate = r_a_map + r_b_map
        D0 = np.array([r_b_map / total_rate, r_a_map / total_rate]) if total_rate > 0 else np.array([0.5, 0.5])
        v = np.broadcast_to(D0, (n_lam, n_mu, 2)).copy()
        log_scale = np.zeros((n_lam, n_mu))

        for t in range(N):
            ic = count_indices[t]
            R_t = R_all_map[:, :, ic, :, :]
            v = np.einsum('lmij,lmj->lmi', R_t, v)
            scale = v[:, :, 0] + v[:, :, 1]
            mask = scale > 0
            v[mask, 0] /= scale[mask]
            v[mask, 1] /= scale[mask]
            log_scale[mask] += np.log(scale[mask])
            log_scale[~mask] = -np.inf

        final_sum = v[:, :, 0] + v[:, :, 1]
        log_likes_map = np.where(
            (final_sum > 0) & np.isfinite(log_scale),
            log_scale + np.log(np.maximum(final_sum, 1e-300)),
            -np.inf
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
