#!/usr/bin/env python3
"""Diffusion-extension simulation for the DP score-estimation paper.

Pipeline (1D, CPU friendly):

    data -> projected LMC score draw -> DP histogram -> local-affine density
         -> exact OU-smoothed score of that density -> reverse OU sampler.

The parameter draw uses projected Langevin Monte Carlo, matching the practical
mechanism in the paper.  Projection onto the 1D coefficient polytope is done by
Dykstra's algorithm, which converges to the Euclidean projection onto

    |theta_j| <= B,  |theta_{j+1}-theta_j| <= D.

The forward OU score of the cellwise-affine surrogate is evaluated from exact
truncated-Gaussian moments; no score network is trained.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import wasserstein_distance

# Import the core score/density implementation from the simulation suite.
HERE = Path(__file__).resolve().parent
SUITE = HERE / "suite"
if not SUITE.exists():
    # Compatibility fallback when this module is copied next to the suite.
    SUITE = HERE.parent / "dp_score_simulation_suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

from dp_score_sim.core import (  # noqa: E402
    SimConfig, sample_target, build_model, score_from_theta,
    true_score, theta_risk,
)
from dp_score_sim.density import (  # noqa: E402
    target_density, make_uniform_edges, histogram_masses,
    evaluate_local_affine, grid_tv, _safe_slopes,
)


# ---------------------------------------------------------------------------
# Exact Euclidean projection (to numerical tolerance) by Dykstra cycles
# ---------------------------------------------------------------------------

def _project_pair_strip(z0: float, z1: float, D: float) -> tuple[float, float]:
    """Euclidean projection of (z0,z1) onto |z1-z0| <= D."""
    diff = z1 - z0
    if diff > D:
        e = 0.5 * (diff - D)
        return z0 + e, z1 - e
    if diff < -D:
        e = 0.5 * (-D - diff)
        return z0 - e, z1 + e
    return z0, z1


def dykstra_project(
    y: np.ndarray,
    B: float,
    D: float,
    max_cycles: int = 80,
    tol: float = 1e-10,
) -> tuple[np.ndarray, int, float]:
    """Project y onto box + adjacent-difference constraints.

    Cyclic Dykstra projection onto one box and m-1 pairwise strips.  For this
    1D chain polytope the elementary set projections are analytic.
    """
    x = np.asarray(y, dtype=float).copy()
    m = x.size
    p_box = np.zeros(m)
    p_pair = np.zeros((max(m - 1, 0), 2))

    for cyc in range(1, max_cycles + 1):
        x_old = x.copy()

        # Box set.
        z = x + p_box
        xn = np.clip(z, -B, B)
        p_box = z - xn
        x = xn

        # Pairwise strips.
        for j in range(m - 1):
            z0 = x[j] + p_pair[j, 0]
            z1 = x[j + 1] + p_pair[j, 1]
            q0, q1 = _project_pair_strip(z0, z1, D)
            p_pair[j, 0] = z0 - q0
            p_pair[j, 1] = z1 - q1
            x[j], x[j + 1] = q0, q1

        change = float(np.max(np.abs(x - x_old)))
        box_v = max(0.0, float(np.max(np.abs(x)) - B))
        diff_v = 0.0 if m <= 1 else max(0.0, float(np.max(np.abs(np.diff(x))) - D))
        violation = max(box_v, diff_v)
        if change <= tol and violation <= 10 * tol:
            return x, cyc, violation

    box_v = max(0.0, float(np.max(np.abs(x)) - B))
    diff_v = 0.0 if m <= 1 else max(0.0, float(np.max(np.abs(np.diff(x))) - D))
    return x, max_cycles, max(box_v, diff_v)


@dataclass
class LMCInfo:
    step_size: float
    lipschitz: float
    ridge: float
    n_steps: int
    mean_projection_cycles: float
    max_projection_violation: float
    risk_trace_steps: list[int]
    risk_trace: list[float]


def projected_lmc_draw(
    model,
    rng: np.random.Generator,
    n_steps: int,
    ridge: float,
    step_factor: float,
    x_test: np.ndarray | None = None,
    cfg: SimConfig | None = None,
    trace_every: int = 0,
    projection_cycles: int = 80,
    projection_tol: float = 1e-10,
) -> tuple[np.ndarray, LMCInfo]:
    """One projected-LMC draw from the ridge-regularized target.

    eta = step_factor / L, where L is the smooth-potential gradient Lipschitz
    constant.  The starting point theta=0 is data independent.
    """
    lam_max = float(np.linalg.eigvalsh(model.A).max())
    Lgrad = model.c * lam_max + ridge
    eta = step_factor / max(Lgrad, 1e-14)

    theta = np.zeros(model.h.size)
    proj_cycles_used = []
    max_v = 0.0
    trace_steps, trace_risks = [], []

    for k in range(1, n_steps + 1):
        grad = model.c * (model.A @ theta + model.h) + ridge * theta
        proposal = theta - eta * grad + math.sqrt(2.0 * eta) * rng.normal(size=theta.size)
        theta, cyc, v = dykstra_project(
            proposal,
            model.coef_bound,
            model.diff_bound,
            max_cycles=projection_cycles,
            tol=projection_tol,
        )
        proj_cycles_used.append(cyc)
        max_v = max(max_v, v)

        if trace_every and (k == 1 or k % trace_every == 0 or k == n_steps):
            trace_steps.append(k)
            if x_test is not None and cfg is not None:
                r = theta_risk(
                    model, theta, x_test, cfg.gamma, cfg.target,
                    cfg.amplitude, cfg.K, cfg.lam, cfg.mixture_mu,
                )
                trace_risks.append(float(r))

    info = LMCInfo(
        step_size=eta,
        lipschitz=Lgrad,
        ridge=ridge,
        n_steps=n_steps,
        mean_projection_cycles=float(np.mean(proj_cycles_used)),
        max_projection_violation=float(max_v),
        risk_trace_steps=trace_steps,
        risk_trace=trace_risks,
    )
    return theta, info


# ---------------------------------------------------------------------------
# Cellwise-affine surrogate and exact OU moments
# ---------------------------------------------------------------------------

@dataclass
class AffineCells:
    left: np.ndarray
    right: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    masses: np.ndarray
    slopes: np.ndarray
    intercept: np.ndarray  # density = intercept + slopecoef * y
    slopecoef: np.ndarray


def affine_cells_from_hist(hist, slopes: np.ndarray) -> AffineCells:
    safe = _safe_slopes(np.asarray(slopes, dtype=float), hist.widths)
    base = hist.masses / hist.widths
    # base * [1 + safe*(y-c)] = base*(1-safe*c) + base*safe*y
    intercept = base * (1.0 - safe * hist.centers)
    slopecoef = base * safe
    return AffineCells(
        left=hist.edges[:-1].copy(),
        right=hist.edges[1:].copy(),
        centers=hist.centers.copy(),
        widths=hist.widths.copy(),
        masses=hist.masses.copy(),
        slopes=safe,
        intercept=intercept,
        slopecoef=slopecoef,
    )


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def ou_affine_density_score(
    x: np.ndarray,
    cells: AffineCells,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact P_t rho and score for a 1D cellwise-affine density.

    Forward OU: X_t = a Y + sqrt(v) Z, a=e^{-t}, v=1-e^{-2t}.
    Each cell integral is reduced to truncated Normal moments I0,I1,I2.
    The score uses
        grad log p_t(x) = -(x-a E[Y|X_t=x])/v.
    """
    if t <= 0:
        raise ValueError("t must be positive")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    a = math.exp(-t)
    v = max(1.0 - math.exp(-2.0 * t), 1e-14)
    sigma = math.sqrt(v) / a
    mu = x[:, None] / a

    zl = (cells.left[None, :] - mu) / sigma
    zu = (cells.right[None, :] - mu) / sigma
    cdf = ndtr(zu) - ndtr(zl)
    pl = _phi(zl)
    pu = _phi(zu)

    inva = 1.0 / a
    I0 = inva * cdf
    I1 = inva * (mu * cdf + sigma * (pl - pu))
    I2 = inva * (
        (mu * mu + sigma * sigma) * cdf
        + 2.0 * mu * sigma * (pl - pu)
        + sigma * sigma * (zl * pl - zu * pu)
    )

    A = cells.intercept[None, :]
    B = cells.slopecoef[None, :]
    p = np.sum(A * I0 + B * I1, axis=1)
    ymoment = np.sum(A * I1 + B * I2, axis=1)
    p_safe = np.maximum(p, 1e-300)
    post_mean = ymoment / p_safe
    score = -(x - a * post_mean) / v
    return p_safe, score


def sample_affine_density(cells: AffineCells, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact rejection sampling cell-by-cell from the piecewise affine density."""
    idx = rng.choice(cells.masses.size, size=n, p=cells.masses)
    out = np.empty(n)
    for j in np.unique(idx):
        positions = np.flatnonzero(idx == j)
        need = positions.size
        l, r, c, s = cells.left[j], cells.right[j], cells.centers[j], cells.slopes[j]
        M = 1.0 + abs(s) * 0.5 * (r - l)
        vals = []
        got = 0
        while got < need:
            m = max(64, int(1.7 * (need - got)))
            y = rng.uniform(l, r, size=m)
            f = 1.0 + s * (y - c)
            keep = rng.uniform(size=m) < (f / M)
            acc = y[keep]
            take = min(need - got, acc.size)
            if take:
                vals.append(acc[:take]); got += take
        out[positions] = np.concatenate(vals)
    return out


def sample_ptau(cells: AffineCells, tau: float, n: int, rng: np.random.Generator) -> np.ndarray:
    y0 = sample_affine_density(cells, n, rng)
    a = math.exp(-tau)
    sd = math.sqrt(max(1.0 - math.exp(-2.0 * tau), 0.0))
    return a * y0 + sd * rng.normal(size=n)


# ---------------------------------------------------------------------------
# Adaptive reverse OU grid and sampler
# ---------------------------------------------------------------------------

def adaptive_forward_times(T: float, tau: float, kappa: float) -> np.ndarray:
    """Descending forward-time grid t_0=T > ... > t_N=tau.

    Uses h_k <= kappa * min(1,t_k), the standard shape needed near the
    early-stopping time: constant step away from zero and geometric refinement
    when t<1.  This gives N = O(kappa^{-1}(T+log(1/tau))).
    """
    if not (0 < tau < T):
        raise ValueError("need 0 < tau < T")
    if not (0 < kappa < 1):
        raise ValueError("kappa must lie in (0,1)")
    times = [float(T)]
    t = float(T)
    while t > tau * (1 + 1e-14):
        h = kappa * min(1.0, t)
        h = min(h, t - tau)
        if h <= 0:
            break
        t -= h
        times.append(float(t))
        if len(times) > 2_000_000:
            raise RuntimeError("adaptive grid unexpectedly large")
    times[-1] = float(tau)
    return np.asarray(times)


def reverse_ou_adaptive_samples(
    cells: AffineCells,
    T: float,
    tau: float,
    kappa: float,
    n_particles: int,
    rng: np.random.Generator,
    score_grid: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Euler-Maruyama reverse OU on an adaptive forward-time grid."""
    times = adaptive_forward_times(T, tau, kappa)
    y = rng.normal(size=n_particles)
    for k in range(len(times) - 1):
        t = times[k]
        h = times[k] - times[k + 1]
        _, sg = ou_affine_density_score(score_grid, cells, t)
        sval = np.interp(y, score_grid, sg, left=sg[0], right=sg[-1])
        y = y + h * (y + 2.0 * sval) + math.sqrt(2.0 * h) * rng.normal(size=n_particles)
    return y, len(times) - 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binned_tv_samples_vs_density(
    samples: np.ndarray,
    density_fn,
    xmin: float,
    xmax: float,
    bins: int,
    quad_per_bin: int = 7,
) -> float:
    """Histogram TV proxy using per-bin quadrature for the target mass."""
    edges = np.linspace(xmin, xmax, bins + 1)
    counts, _ = np.histogram(samples, bins=edges)
    emp = counts / len(samples)

    target_mass = np.zeros(bins)
    for j in range(bins):
        xs = np.linspace(edges[j], edges[j + 1], quad_per_bin)
        vals = density_fn(xs)
        target_mass[j] = np.trapezoid(vals, xs)
    inside = float(target_mass.sum())
    tail = max(0.0, 1.0 - inside)
    return float(0.5 * np.sum(np.abs(emp - target_mass)) + 0.5 * tail)


def empirical_tv_between_samples(a: np.ndarray, b: np.ndarray, xmin: float, xmax: float, bins: int) -> float:
    edges = np.linspace(xmin, xmax, bins + 1)
    ca, _ = np.histogram(a, bins=edges)
    cb, _ = np.histogram(b, bins=edges)
    pa = ca / max(len(a), 1)
    pb = cb / max(len(b), 1)
    return float(0.5 * np.sum(np.abs(pa - pb)))


def delta_theory(n: int, alpha: float, gamma: float, d: int = 1) -> float:
    a = n ** (-(gamma + 1.0) / (2.0 * gamma + d + 2.0))
    b = (n * alpha) ** (-(gamma + 1.0) / (d + gamma + 1.0))
    return float(max(a, b))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------





