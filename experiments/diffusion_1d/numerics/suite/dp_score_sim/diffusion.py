"""Lightweight 1D OU post-processing utilities.

No neural network is trained.  The score of P_t rho is computed by direct
Gaussian convolution on a grid, then interpolated inside an Euler reverse-OU
sampler.  This is intended as a numerical proof-of-concept, not as the exact
adaptive-grid implementation used in the theorem.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import wasserstein_distance


def ou_smoothed_density_and_score(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    rho_y: np.ndarray,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute P_t rho and its score by direct quadrature.

    Forward OU convention: dX_t = -X_t dt + sqrt(2) dB_t.
    Thus X_t | X_0=y ~ N(exp(-t)y, 1-exp(-2t)).
    """
    if t <= 0:
        raise ValueError("t must be positive")
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    rho_y = np.asarray(rho_y, dtype=float)

    a = math.exp(-t)
    var = max(1.0 - math.exp(-2.0 * t), 1e-12)
    sd = math.sqrt(var)

    # trapezoidal quadrature weights on y-grid
    w = np.empty_like(y_grid)
    w[1:-1] = 0.5 * (y_grid[2:] - y_grid[:-2])
    w[0] = 0.5 * (y_grid[1] - y_grid[0])
    w[-1] = 0.5 * (y_grid[-1] - y_grid[-2])
    mass_weights = rho_y * w

    diff = x_grid[:, None] - a * y_grid[None, :]
    ker = np.exp(-0.5 * diff**2 / var) / math.sqrt(2.0 * math.pi * var)
    p = ker @ mass_weights
    dp = ((-diff / var) * ker) @ mass_weights
    p = np.maximum(p, 1e-300)
    score = dp / p
    return p, score


def reverse_ou_samples(
    n_particles: int,
    T: float,
    tau: float,
    n_steps: int,
    x_score_grid: np.ndarray,
    y_support_grid: np.ndarray,
    rho_y: np.ndarray,
    rng: np.random.Generator,
    init: str = "gaussian",
) -> np.ndarray:
    """Euler simulation of the reverse OU process from T down to tau.

    Reverse-time drift under the forward OU convention is
        y + 2 * score_t(y),
    where t is the corresponding forward time.
    """
    if not (0 < tau < T):
        raise ValueError("require 0 < tau < T")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    if init != "gaussian":
        raise ValueError("only Gaussian initialization is implemented")
    y = rng.normal(size=n_particles)

    # Reverse clock r in [0, T-tau], forward time t=T-r.
    dt = (T - tau) / n_steps
    for k in range(n_steps):
        t = T - k * dt
        _, score_grid = ou_smoothed_density_and_score(
            x_score_grid, y_support_grid, rho_y, t
        )
        score = np.interp(
            y,
            x_score_grid,
            score_grid,
            left=score_grid[0],
            right=score_grid[-1],
        )
        y = y + dt * (y + 2.0 * score) + math.sqrt(2.0 * dt) * rng.normal(size=n_particles)
    return y


def histogram_tv_proxy(
    samples: np.ndarray,
    true_density_fn,
    x_min: float,
    x_max: float,
    n_bins: int = 120,
) -> float:
    """Binned TV proxy between generated samples and a known density."""
    edges = np.linspace(x_min, x_max, n_bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    emp_mass = counts / max(counts.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    true_mass = true_density_fn(centers) * widths
    # Account for true tail mass outside the plotting interval.
    inside = true_mass.sum()
    true_mass = np.maximum(true_mass, 0.0)
    tv_inside = 0.5 * np.sum(np.abs(emp_mass - true_mass))
    tail = max(0.0, 1.0 - inside)
    return float(tv_inside + 0.5 * tail)


def wasserstein_to_reference(samples: np.ndarray, reference: np.ndarray) -> float:
    return float(wasserstein_distance(samples, reference))
