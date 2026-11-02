"""Projected Langevin diagnostics for the 1D coefficient polytope.

The paper's practical mechanism uses projected LMC after adding a small ridge.
For a lightweight CPU diagnostic we use repeated pairwise projections onto the
box and neighboring-difference constraints.  This converges to a feasible point
but is not advertised here as an exact Euclidean projection routine; therefore
these utilities are for computational diagnostics only, not privacy accounting.
"""
from __future__ import annotations

import math
import numpy as np

from .core import Model1D


def project_feasible_pairwise(
    theta: np.ndarray,
    B: float,
    D: float,
    passes: int = 25,
) -> np.ndarray:
    """Repeated projection sweeps onto box and adjacent-difference strips."""
    x = np.asarray(theta, dtype=float).copy()
    np.clip(x, -B, B, out=x)
    if x.size <= 1:
        return x

    for _ in range(passes):
        max_violation = 0.0
        # Forward and backward sweeps reduce ordering artifacts.
        for order in (range(x.size - 1), range(x.size - 2, -1, -1)):
            for j in order:
                diff = x[j + 1] - x[j]
                if diff > D:
                    e = 0.5 * (diff - D)
                    x[j] += e
                    x[j + 1] -= e
                    max_violation = max(max_violation, diff - D)
                elif diff < -D:
                    e = 0.5 * (-D - diff)
                    x[j] -= e
                    x[j + 1] += e
                    max_violation = max(max_violation, -D - diff)
            np.clip(x, -B, B, out=x)
        if max_violation < 1e-10:
            break
    return x


def feasible_violation(theta: np.ndarray, B: float, D: float) -> float:
    theta = np.asarray(theta, dtype=float)
    v_box = max(0.0, float(np.max(np.abs(theta)) - B))
    if theta.size <= 1:
        return v_box
    v_diff = max(0.0, float(np.max(np.abs(np.diff(theta))) - D))
    return max(v_box, v_diff)


def projected_langevin_path(
    model: Model1D,
    rng: np.random.Generator,
    n_steps: int,
    step_size: float,
    ridge: float = 1e-3,
    save_every: int = 10,
    projection_passes: int = 25,
) -> tuple[list[int], list[np.ndarray], list[float]]:
    """Projected Langevin path for the ridge-regularized target.

    Potential:
        U(theta) = c[0.5 theta^T A theta + h^T theta]
                   + 0.5*ridge*||theta||^2.

    Update:
        theta <- Proj(theta - eta grad U + sqrt(2 eta) Z).
    """
    m = model.h.size
    theta = np.zeros(m)
    saved_steps: list[int] = []
    saved: list[np.ndarray] = []
    violations: list[float] = []

    for k in range(1, n_steps + 1):
        grad = model.c * (model.A @ theta + model.h) + ridge * theta
        proposal = theta - step_size * grad + math.sqrt(2.0 * step_size) * rng.normal(size=m)
        theta = project_feasible_pairwise(
            proposal, model.coef_bound, model.diff_bound, passes=projection_passes
        )

        if k % save_every == 0 or k == 1 or k == n_steps:
            saved_steps.append(k)
            saved.append(theta.copy())
            violations.append(feasible_violation(theta, model.coef_bound, model.diff_bound))

    return saved_steps, saved, violations
