"""
Bayesian Estimation of Switching Rates for Blinking Emitters
=============================================================

Implementation of the paper:
    Geordy et al., "Bayesian estimation of switching rates for blinking emitters",
    New J. Phys. 21 (2019) 063001

This package provides:
    - simulation: Generate synthetic blinking time traces (DTMC and CTMC)
    - dtmc_single: DTMC single-step Bayesian inference (Section 2)
    - ctmc: CTMC Bayesian inference with Bessel functions (Section 3)
    - dtmc_multi: DTMC multi-step inference with recursion (Section 4)
    - state_inference: Determine underlying hidden states (Section 5)
    - threshold: Conventional threshold analysis for comparison
    - plotting: Visualization utilities for reproducing paper figures
"""
