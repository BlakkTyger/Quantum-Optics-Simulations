# Bayesian Estimation of Switching Rates for Blinking Emitters

**Implementation of:** Geordy et al., *"Bayesian estimation of switching rates for blinking emitters"*, New J. Phys. 21 (2019) 063001  
**arXiv:** [1811.06627](https://arxiv.org/abs/1811.06627)

## Overview

This project provides a complete Python implementation of Bayesian inference methods for determining switching rates of blinking quantum emitters from photon count time-series data. The implementation covers all models and results presented in the paper.

### Physical Problem

Single quantum emitters (diamond colour centres, quantum dots, etc.) exhibit **blinking** — random switching between fluorescent (on) and dark (off) states. Characterizing the switching rates is critical for understanding the underlying physical mechanisms. Traditional threshold analysis is unreliable for:
- Low signal-to-noise ratios
- Fast switching rates (many switches per detector interval)

The Bayesian approach developed in the paper infers switching rates directly from count data without thresholds.

## Models Implemented

### 1. DTMC Single-Step Model (Section 2)
- Emitter state fixed during each detector interval
- Switching only at interval boundaries
- Parameters: α₁ (switch-on probability), β₁ (switch-off probability)
- Efficient computation via 2×2 matrix products

### 2. CTMC Model (Section 3)
- Emitter switches at any continuous time point
- Multiple switches possible per detector interval
- Parameters: rₐ (switch-on rate), r_b (switch-off rate)
- Uses Bessel functions for the fraction-on distribution
- Requires numerical integration (scipy.integrate.quad)

### 3. DTMC Multi-Step Model (Section 4)
- Approximates CTMC by subdividing intervals into d = 2^m subintervals
- Recursion relation for efficient computation via discrete convolutions
- Converges to CTMC as d → ∞

### 4. State Inference (Section 5)
- Forward-backward algorithm to determine P(state=a | all data)
- Provides probability distribution over states (not binary assignment)
- Automatically accounts for parameter uncertainty

### 5. Threshold Analysis (Section 2.3)
- Conventional method for comparison
- Multiple threshold selection methods from the literature
- Demonstrates sensitivity to arbitrary choices

## Project Structure

```
Bayesian-Estimation-Blinking-QD/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Bayesian-Estimation.pdf      # Original paper
├── run_all.py                   # Main script to reproduce all figures
├── src/
│   ├── __init__.py              # Package definition
│   ├── simulation.py            # Data generation (DTMC & CTMC)
│   ├── dtmc_single.py           # DTMC single-step Bayesian inference
│   ├── ctmc.py                  # CTMC Bayesian inference
│   ├── dtmc_multi.py            # DTMC multi-step inference (recursion)
│   ├── state_inference.py       # Hidden state inference
│   ├── threshold.py             # Threshold analysis (comparison)
│   └── plotting.py              # Visualization for all figures
└── figures/                     # Generated figures (created by run_all.py)
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements: numpy, scipy, matplotlib, tqdm

## Usage

### Reproduce All Figures
```bash
python run_all.py              # Full resolution (slow, ~hours)
python run_all.py --fast       # Reduced resolution (faster, ~minutes)
```

### Generate Specific Figures
```bash
python run_all.py --figure 1           # Figure 1 only
python run_all.py --figure 4 5 6       # Figures 4, 5, and 6
python run_all.py --figure 7 --fast    # Figure 7 at reduced resolution
```

### Use as Library
```python
from src.simulation import simulate_dtmc_single_step, simulate_ctmc
from src.dtmc_single import compute_posterior_grid
from src.ctmc import compute_posterior_ctmc
from src.dtmc_multi import compute_posterior_multi
from src.state_inference import infer_states_known_params

# Simulate blinking data
counts, states = simulate_dtmc_single_step(
    alpha=0.05, beta=0.03, lam=30.0, mu=5.0, N=500, seed=42
)

# Perform Bayesian inference
result = compute_posterior_grid(
    counts,
    alpha_range=(0.001, 0.2),
    beta_range=(0.001, 0.15),
    lam_known=30.0, mu_known=5.0,
    n_alpha=100, n_beta=100
)

# Access posterior distribution
posterior = result['posterior']        # 2D normalized posterior
alpha_map = result['alpha_grid'][...]  # MAP estimate
```

## Figures Reproduced

| Figure | Description | Model |
|--------|-------------|-------|
| 1 | Blinking time traces (clear, noisy, fast-switching) | Simulation |
| 4 | DTMC inference on noisy data (credible regions + marginals) | DTMC single |
| 5 | Inference with high switching probability (α=0.8, β=0.9) | DTMC single |
| 6 | Bayesian vs threshold analysis comparison | DTMC single + Threshold |
| 7 | CTMC inference on fast-switching data | CTMC |
| 8 | Accuracy map: single-step model on CTMC-generated data | DTMC single |
| 9 | Multi-step error vs number of subintervals d | DTMC multi |
| 10 | DTMC multi-step inference (d=16) on fast-switching data | DTMC multi |
| 11 | State inference with convergence demonstration | State inference |

## Mathematical Summary

### Key Equation: Matrix Product Likelihood

The likelihood of observing counts **c** = (c₁, ..., c_N) is computed efficiently as:

```
P(c | Ω) = [1, 1] · R_N · R_{N-1} · ... · R_1 · D₀
```

where each R_t is a 2×2 transfer matrix:

```
R_t = [[P(c_t, s_t=0 | s_{t-1}=0),  P(c_t, s_t=0 | s_{t-1}=1)],
       [P(c_t, s_t=1 | s_{t-1}=0),  P(c_t, s_t=1 | s_{t-1}=1)]]
```

This converts an exponential-complexity sum over 2^N state sequences into an O(N) matrix product.

### CTMC: Bessel Function Solutions

For the continuous-time model, the distribution R_{ab}(f) over the fraction f spent in the on state involves modified Bessel functions I₀ and I₁:

```
R₀₁(f) = rₐ · I₀(2√(f(1-f)·rₐ·r_b)) · exp(-rₐ(1-f) - r_b·f)
R₁₀(f) = r_b · I₀(2√(f(1-f)·rₐ·r_b)) · exp(-rₐ(1-f) - r_b·f)
```

### Multi-Step Recursion

The recursion relation for doubling the interval:

```
f_{i,j}(τ, k) = Σ_z Σ_{c=0}^{k} f_{i,z}(τ/2, c) · f_{z,j}(τ/2, k-c)
```

This is a discrete convolution summed over intermediate states, applied m = log₂(d) times.

## Computational Notes

- **Underflow prevention**: Matrix products are rescaled periodically with log-scale accumulation.
- **Grid-based integration**: Posterior is evaluated on a discrete grid; marginalisation uses trapezoidal approximation.
- **CTMC numerical integration**: Uses scipy's adaptive quadrature (QUADPACK) for the fraction integral.
- **Convolution**: Multi-step recursion uses numpy's `convolve` for discrete convolutions.

## Citation

```bibtex
@article{geordy2019bayesian,
  title={Bayesian estimation of switching rates for blinking emitters},
  author={Geordy, Jemy and Rogers, Lachlan J and Rogers, Cameron M and Volz, Thomas and Gilchrist, Alexei},
  journal={New Journal of Physics},
  volume={21},
  number={6},
  pages={063001},
  year={2019},
  publisher={IOP Publishing}
}
```
