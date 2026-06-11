"""
Simulation Module for Blinking Quantum Emitters
================================================

This module generates synthetic photon count time-series data for blinking
quantum emitters using two different stochastic models:

1. DTMC (Discrete Time Markov Chain) Single-Step Model:
   - The emitter state (on/off) is fixed for each detector interval.
   - Switching can only occur at interval boundaries.
   - Switch-on probability: alpha, Switch-off probability: beta.

2. CTMC (Continuous Time Markov Chain) Model:
   - The emitter can switch states at any continuous time point.
   - Switch-on rate: r_a, Switch-off rate: r_b.
   - Multiple switches possible within a single detector interval.

Physics:
   - On state: photon counts ~ Poisson(mu + lambda) where lambda is fluorescence rate.
   - Off state: photon counts ~ Poisson(mu) where mu is background/dark count rate.
   - Combined rate for on state is (mu + lambda) since background noise is always present.

References:
   - Geordy et al., New J. Phys. 21 (2019) 063001, Sections 2.1 and 3.1
"""

import numpy as np
from typing import Tuple, Optional

from .cache import disk_cache

@disk_cache()
def simulate_dtmc_single_step(
    alpha: float,
    beta: float,
    lam: float,
    mu: float,
    N: int,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a blinking time trace using the DTMC single-step model.

    In this model, the emitter state is fixed for the entire detector interval
    and can only switch at the boundaries of the intervals.

    Parameters
    ----------
    alpha : float
        Probability of switching from off (0) to on (1) at each interval boundary.
        Must be in [0, 1].
    beta : float
        Probability of switching from on (1) to off (0) at each interval boundary.
        Must be in [0, 1].
    lam : float
        Fluorescence rate (expected photon counts per interval when on).
        The total rate when on is (mu + lam).
    mu : float
        Background/dark count rate (expected counts per interval when off).
    N : int
        Number of detector intervals (time steps) to simulate.
    initial_state : int, optional
        Starting state (0=off, 1=on). If None, drawn from stationary distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    counts : np.ndarray of shape (N,)
        Simulated photon counts at each detector interval.
    states : np.ndarray of shape (N,)
        True underlying state (0=off, 1=on) at each interval.

    Notes
    -----
    The transition matrix for the DTMC is:
        P(s_t=1 | s_{t-1}=0) = alpha  (switch on)
        P(s_t=0 | s_{t-1}=1) = beta   (switch off)
        P(s_t=0 | s_{t-1}=0) = 1 - alpha
        P(s_t=1 | s_{t-1}=1) = 1 - beta

    Count distribution:
        c_t | s_{t-1}=0 ~ Poisson(mu)
        c_t | s_{t-1}=1 ~ Poisson(mu + lam)
    """
    rng = np.random.default_rng(seed)

    # Initialize state from stationary distribution if not specified
    if initial_state is None:
        # Stationary distribution: pi_on = alpha / (alpha + beta)
        pi_on = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
        initial_state = 1 if rng.random() < pi_on else 0

    states = np.zeros(N, dtype=int)
    counts = np.zeros(N, dtype=int)

    # Set initial state
    current_state = initial_state

    for t in range(N):
        # Record the state at the beginning of this interval
        states[t] = current_state

        # Generate counts based on current state
        if current_state == 0:
            counts[t] = rng.poisson(mu)
        else:
            counts[t] = rng.poisson(mu + lam)

        # Transition to next state at the boundary
        if current_state == 0:
            # Off state: switch on with probability alpha
            if rng.random() < alpha:
                current_state = 1
        else:
            # On state: switch off with probability beta
            if rng.random() < beta:
                current_state = 0

    return counts, states

@disk_cache()
def simulate_ctmc(
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    N: int,
    T: float = 1.0,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate a blinking time trace using the Continuous Time Markov Chain (CTMC) model.

    In this model, the emitter can switch states at any continuous time point.
    The time spent in the off state is exponentially distributed with rate r_a
    (switch-on rate), and time in the on state with rate r_b (switch-off rate).

    Parameters
    ----------
    r_a : float
        Rate of switching from off (0) to on (1). Units: per detector interval.
        Mean time in off state = 1/r_a.
    r_b : float
        Rate of switching from on (1) to off (0). Units: per detector interval.
        Mean time in on state = 1/r_b.
    lam : float
        Fluorescence rate (expected photon counts per detector interval when on).
    mu : float
        Background/dark count rate (expected counts per detector interval).
    N : int
        Number of detector intervals to simulate.
    T : float
        Duration of each detector interval (default=1.0, as in paper).
    initial_state : int, optional
        Starting state (0=off, 1=on). If None, drawn from stationary distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    counts : np.ndarray of shape (N,)
        Simulated photon counts at each detector interval.
    states_at_boundaries : np.ndarray of shape (N+1,)
        State at the boundary of each interval (including initial state s_0).
    fractions_on : np.ndarray of shape (N,)
        Fraction of each interval spent in the on state.

    Notes
    -----
    The CTMC is governed by the rate matrix:
        Q = [[-r_a,  r_a],
             [ r_b, -r_b]]

    The transition probabilities from state a to state b in time t are:
        P_00(t) = (r_b + r_a * exp(-(r_a+r_b)*t)) / (r_a + r_b)
        P_01(t) = r_a * (1 - exp(-(r_a+r_b)*t)) / (r_a + r_b)
        P_10(t) = r_b * (1 - exp(-(r_a+r_b)*t)) / (r_a + r_b)
        P_11(t) = (r_a + r_b * exp(-(r_a+r_b)*t)) / (r_a + r_b)

    Within each interval, counts are Poisson with rate:
        rate = mu * (1 - f) + (mu + lam) * f = mu + lam * f
    where f is the fraction of the interval spent in the on state.
    """
    rng = np.random.default_rng(seed)

    # Initialize state from stationary distribution if not specified
    if initial_state is None:
        # Stationary distribution: pi_on = r_a / (r_a + r_b)
        pi_on = r_a / (r_a + r_b) if (r_a + r_b) > 0 else 0.5
        initial_state = 1 if rng.random() < pi_on else 0

    counts = np.zeros(N, dtype=int)
    states_at_boundaries = np.zeros(N + 1, dtype=int)
    fractions_on = np.zeros(N)

    states_at_boundaries[0] = initial_state
    current_state = initial_state

    for n in range(N):
        # Simulate the CTMC within this detector interval [0, T]
        time_in_interval = 0.0
        time_on = 0.0
        state = current_state

        while time_in_interval < T:
            # Time to next switch is exponentially distributed
            if state == 0:
                # In off state, rate of leaving is r_a
                wait_time = rng.exponential(1.0 / r_a) if r_a > 0 else np.inf
            else:
                # In on state, rate of leaving is r_b
                wait_time = rng.exponential(1.0 / r_b) if r_b > 0 else np.inf

            if time_in_interval + wait_time >= T:
                # No switch before end of interval
                remaining = T - time_in_interval
                if state == 1:
                    time_on += remaining
                time_in_interval = T
            else:
                # Switch occurs
                if state == 1:
                    time_on += wait_time
                time_in_interval += wait_time
                state = 1 - state  # Toggle state

        # Record fraction of interval spent on
        fractions_on[n] = time_on / T

        # Record state at end of interval
        current_state = state
        states_at_boundaries[n + 1] = current_state

        # Generate counts: rate = mu + lam * fraction_on
        effective_rate = mu + lam * fractions_on[n]
        counts[n] = rng.poisson(effective_rate)

    return counts, states_at_boundaries, fractions_on

@disk_cache()
def simulate_dtmc_multi_step(
    r_a: float,
    r_b: float,
    lam: float,
    mu: float,
    N: int,
    d: int = 4,
    initial_state: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a blinking time trace using the DTMC multi-step model.

    The detector interval is divided into d subintervals. The emitter can
    switch state only at subinterval boundaries, with switching probabilities
    derived from the CTMC rates.

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
    N : int
        Number of detector intervals.
    d : int
        Number of subintervals per detector interval (should be power of 2).
    initial_state : int, optional
        Starting state. If None, drawn from stationary distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    counts : np.ndarray of shape (N,)
        Simulated photon counts at each detector interval.
    states_at_boundaries : np.ndarray of shape (N+1,)
        State at the boundary of each detector interval.

    Notes
    -----
    The switching probabilities over a subinterval of duration T/d are:
        alpha_d = 1 - exp(-r_a * T / d)    [Eq. 43]
        beta_d  = 1 - exp(-r_b * T / d)    [Eq. 44]

    Counts in each subinterval are:
        off state: Poisson(mu / d)
        on state:  Poisson((mu + lam) / d)
    """
    rng = np.random.default_rng(seed)

    # Switching probabilities per subinterval (Equations 43, 44)
    alpha_d = 1.0 - np.exp(-r_a / d)
    beta_d = 1.0 - np.exp(-r_b / d)

    # Rates per subinterval
    mu_sub = mu / d
    lam_sub = lam / d

    # Initialize state
    if initial_state is None:
        pi_on = r_a / (r_a + r_b) if (r_a + r_b) > 0 else 0.5
        initial_state = 1 if rng.random() < pi_on else 0

    counts = np.zeros(N, dtype=int)
    states_at_boundaries = np.zeros(N + 1, dtype=int)
    states_at_boundaries[0] = initial_state
    current_state = initial_state

    for n in range(N):
        interval_counts = 0
        state = current_state

        for _ in range(d):
            # Generate counts for this subinterval
            if state == 0:
                interval_counts += rng.poisson(mu_sub)
            else:
                interval_counts += rng.poisson(mu_sub + lam_sub)

            # Transition at subinterval boundary
            if state == 0:
                if rng.random() < alpha_d:
                    state = 1
            else:
                if rng.random() < beta_d:
                    state = 0

        counts[n] = interval_counts
        current_state = state
        states_at_boundaries[n + 1] = current_state

    return counts, states_at_boundaries


def generate_figure1_data(seed: int = 42) -> dict:
    """
    Generate the three example time traces shown in Figure 1 of the paper.

    Figure 1(a): Clear blinking with well-separated on/off states.
    Figure 1(b): Higher noise, same switching rates, harder to threshold.
    Figure 1(c): High switching rate relative to detector interval.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    data : dict
        Dictionary containing 'fig1a', 'fig1b', 'fig1c' sub-dicts,
        each with 'counts', 'states', and 'params'.
    """
    # Figure 1(a): Clear blinking, high signal-to-noise
    # Low switching rates, high signal — threshold analysis works here
    # Paper uses N=10000 for all panels
    params_a = {'alpha': 0.01, 'beta': 0.005, 'lam': 30.0, 'mu': 5.0, 'N': 10000}
    counts_a, states_a = simulate_dtmc_single_step(
        **params_a, seed=seed
    )

    # Figure 1(b): Same switching rates but noisier (lower lambda)
    # On and off count distributions overlap — threshold fails
    params_b = {'alpha': 0.01, 'beta': 0.005, 'lam': 8.0, 'mu': 5.0, 'N': 10000}
    counts_b, states_b = simulate_dtmc_single_step(
        **params_b, seed=seed + 1
    )

    # Figure 1(c): High switching rates (CTMC model needed)
    # Many switches per detector interval — can't even see blinking
    params_c = {'r_a': 3.0, 'r_b': 2.0, 'lam': 10.0, 'mu': 3.0, 'N': 10000}
    counts_c, boundaries_c, fractions_c = simulate_ctmc(
        **params_c, seed=seed + 2
    )

    data = {
        'fig1a': {
            'counts': counts_a,
            'states': states_a,
            'params': params_a,
            'model': 'DTMC_single'
        },
        'fig1b': {
            'counts': counts_b,
            'states': states_b,
            'params': params_b,
            'model': 'DTMC_single'
        },
        'fig1c': {
            'counts': counts_c,
            'states_at_boundaries': boundaries_c,
            'fractions_on': fractions_c,
            'params': params_c,
            'model': 'CTMC'
        }
    }

    return data
