#!/usr/bin/env python3
"""One-dimensional spline score-estimation model and sampling helpers.

Provides projected_lmc_draw and one_replicate (one final LMC release).
Legacy Gibbs helpers remain for comparison. The accompanying experiment_common.py
uses this same projected-LMC kernel for convergence and risk studies.

The old standalone rate and privacy-phase experiments and their CLI have
been removed. Running this file directly does not start an experiment.

Dependencies: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import minimize, Bounds, LinearConstraint


# ============================================================
# Synthetic targets
#
# Default: a multiscale Holder score perturbation
#
#   log rho(x) = const - x^2/2
#                - a sum_{k=0}^K 2^{-k(gamma+1)} cos(2^k x)
#
# so
#
#   s0(x) = -x + a sum_{k=0}^K 2^{-k gamma} sin(2^k x).
#
# For 0 < gamma < 1, the oscillatory part has a gamma-Holder
# modulus uniformly in K.  This is much closer to a multiscale
# worst-case Holder function than the single-cusp example.
#
# A legacy single-cusp target is kept for comparison:
#   rho(x) ∝ exp(-x^2/2 - lam |x|^(1+gamma)).
# ============================================================

def multiscale_score(
    x: np.ndarray,
    gamma: float,
    amplitude: float,
    K: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = -x.copy()
    for k in range(K + 1):
        out += (
            amplitude
            * 2.0 ** (-k * gamma)
            * np.sin((2.0**k) * x)
        )
    return out


def cusp_score(
    x: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return -x - lam * (1.0 + gamma) * np.sign(x) * np.abs(x) ** gamma


def true_score(
    x: np.ndarray,
    gamma: float,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 40,
    lam: float = 0.25,
) -> np.ndarray:
    if target == "multiscale":
        return multiscale_score(x, gamma, amplitude, K)
    if target == "cusp":
        return cusp_score(x, gamma, lam)
    raise ValueError(f"unknown target: {target}")


def multiscale_holder_constant(
    gamma: float,
    amplitude: float,
    K: int,
) -> float:
    """
    A safe local Holder constant for the full score on |x-y| <= 1.

    For 0 < gamma < 1, the bound is uniform in K.  At gamma=1,
    the finite series is Lipschitz with constant <= 1+a(K+1).
    """
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError("gamma must lie in (0,1]")

    if gamma == 1.0:
        return 1.0 + amplitude * (K + 1)

    low = 2.0 ** (1.0 - gamma) / (
        2.0 ** (1.0 - gamma) - 1.0
    )
    high = 2.0 / (1.0 - 2.0 ** (-gamma))
    return 1.0 + amplitude * (low + high)


def score_sup_bound(
    xmax: float,
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
) -> float:
    if target == "multiscale":
        geom = sum(2.0 ** (-k * gamma) for k in range(K + 1))
        return xmax + amplitude * geom
    if target == "cusp":
        return xmax + lam * (1.0 + gamma) * xmax**gamma
    raise ValueError(f"unknown target: {target}")


def holder_constant(
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
) -> float:
    if target == "multiscale":
        return multiscale_holder_constant(gamma, amplitude, K)
    if target == "cusp":
        return 1.0 + 2.0 * lam * (1.0 + gamma)
    raise ValueError(f"unknown target: {target}")


def sample_target(
    n: int,
    gamma: float,
    rng: np.random.Generator,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 40,
    lam: float = 0.25,
) -> np.ndarray:
    """Exact rejection samplers using N(0,1) as proposal."""
    chunks = []
    need = n

    if target == "multiscale":
        # rho / phi ∝ exp[-a C(x)], where
        # C(x)=sum 2^{-k(gamma+1)} cos(2^k x).
        # Since |C(x)| <= S, exp[-a C(x)] <= exp(a S).
        S = sum(
            2.0 ** (-k * (gamma + 1.0))
            for k in range(K + 1)
        )

        while need > 0:
            m = max(1024, int(2.4 * need))
            x = rng.normal(size=m)
            C = np.zeros_like(x)
            for k in range(K + 1):
                C += (
                    2.0 ** (-k * (gamma + 1.0))
                    * np.cos((2.0**k) * x)
                )
            accept_prob = np.exp(-amplitude * (C + S))
            accept = rng.uniform(size=m) < accept_prob
            accepted = x[accept]
            take = min(need, accepted.size)
            if take > 0:
                chunks.append(accepted[:take])
                need -= take

        return np.concatenate(chunks)

    if target == "cusp":
        while need > 0:
            m = max(1024, int(2.2 * need))
            x = rng.normal(size=m)
            accept = rng.uniform(size=m) < np.exp(
                -lam * np.abs(x) ** (1.0 + gamma)
            )
            accepted = x[accept]
            take = min(need, accepted.size)
            if take > 0:
                chunks.append(accepted[:take])
                need -= take
        return np.concatenate(chunks)

    raise ValueError(f"unknown target: {target}")


# ============================================================
# Quadratic cardinal B-spline from the paper
# ============================================================

def _pos(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def B2(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return (
        0.5 * _pos(u) ** 2
        - 1.5 * _pos(u - 1.0) ** 2
        + 1.5 * _pos(u - 2.0) ** 2
        - 0.5 * _pos(u - 3.0) ** 2
    )


def B2_prime(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return (
        _pos(u)
        - 3.0 * _pos(u - 1.0)
        + 3.0 * _pos(u - 2.0)
        - _pos(u - 3.0)
    )


def spline_indices(R: float, b: float) -> np.ndarray:
    """
    Indices whose B-spline support can intersect [-2R, 2R].
    Since supp psi_j = [bj, b(j+3)].
    """
    j_min = math.floor(-2.0 * R / b) - 3
    j_max = math.ceil(2.0 * R / b)
    return np.arange(j_min, j_max + 1, dtype=int)


def spline_design(
    x: np.ndarray,
    idx: np.ndarray,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns Psi and dPsi/dx.
    """
    x = np.asarray(x, dtype=float)
    u = x[:, None] / b - idx[None, :]
    psi = B2(u)
    dpsi = B2_prime(u) / b
    return psi, dpsi


def cutoff_chi(x: np.ndarray, R: float) -> np.ndarray:
    """
    A simple C^1 implementation choice:
      chi=1 on [-R,R],
      chi=0 outside [-2R,2R],
      cubic smoothstep in between.

    The draft only requires a smooth cutoff with these properties.
    """
    x = np.asarray(x, dtype=float)
    a = np.abs(x)
    out = np.ones_like(a)

    out[a >= 2.0 * R] = 0.0
    mid = (a > R) & (a < 2.0 * R)

    u = (a[mid] - R) / R
    out[mid] = 1.0 - 3.0 * u**2 + 2.0 * u**3
    return out


# ============================================================
# Exponential-mechanism target
# ============================================================

@dataclass
class Model1D:
    idx: np.ndarray
    b: float
    b_raw: float
    b_capped: bool
    R: float
    coef_bound: float
    diff_bound: float
    holder_L: float
    Q_box: float
    Q_diff: float
    Q: float
    c: float
    A: np.ndarray
    h: np.ndarray
    n: int
    gamma: float
    alpha: float | None


def build_model(
    x: np.ndarray,
    gamma: float,
    alpha: float | None,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 40,
    lam: float = 0.25,
    bandwidth_factor: float = 0.50,
    R_factor: float = 1.25,
    coef_bound_factor: float = 1.10,
) -> Model1D:
    """
    Build the 1D exponential-mechanism target

        pi_D(theta) ∝ 1{theta in Theta}
                      exp[- alpha/(4Q) * sum_i ell_theta(Z_i)].

    On the truncated region, chi_R=1 and chi'_R=0, hence

        sum_i ell_theta(Z_i)
        = 1/2 theta^T A theta + h^T theta.

    The multiplicative constants in b_n and R_n are not identified by
    the asymptotic theorem.  bandwidth_factor and coef_bound_factor are
    finite-sample calibration constants; keep them fixed across all n
    and alpha in a reported experiment.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    d = 1

    # Safe local Holder constant for the selected synthetic score.
    holder_L = holder_constant(
        gamma=gamma,
        target=target,
        amplitude=amplitude,
        K=K,
        lam=lam,
    )

    # R ~ sqrt(log n), as in the theory.
    R = max(2.5, R_factor * math.sqrt(math.log(n + 1.0)))

    # The theorem's bandwidth order.  For alpha=None we use the
    # non-private bandwidth only; this is used for the constrained
    # score-matching ERM diagnostic.
    nonprivate_b = n ** (-1.0 / (d + 2.0 * gamma + 2.0))
    if alpha is None or not np.isfinite(alpha):
        private_b = 0.0
    else:
        private_b = (n * alpha) ** (-1.0 / (d + gamma + 1.0))

    b_raw = bandwidth_factor * max(nonprivate_b, private_b) * math.sqrt(
        math.log(n + 1.0)
    )

    # For d=1 the approximation lemma assumes b <= 1/3.
    b = min(b_raw, 0.30)
    b_capped = bool(b_raw > 0.30 + 1e-14)

    idx = spline_indices(R, b)

    # The theorem uses a deterministic class-dependent coefficient bound.
    # In this synthetic family we can use a sharp known-family bound on
    # the true score over the relevant basis-support region.
    xmax = 2.0 * R + 3.0 * b
    sup_bound = score_sup_bound(
        xmax=xmax,
        gamma=gamma,
        target=target,
        amplitude=amplitude,
        K=K,
        lam=lam,
    )
    coef_bound = coef_bound_factor * sup_bound

    # Neighboring coefficients satisfy |theta_{j+1}-theta_j| <= L b^gamma.
    diff_bound = holder_L * b**gamma

    # Truncation operator from the paper:
    # T_R(x) = x inside [-R,R], and 0 outside.
    z = np.where(np.abs(x) <= R, x, 0.0)

    psi, dpsi = spline_design(z, idx, b)

    # Sum of score-matching losses:
    # 1/2 theta^T A theta + h^T theta.
    A = psi.T @ psi
    h = dpsi.sum(axis=0)

    # 1D version of Q_{n,gamma}, with its two components saved
    # separately so the finite-sample sensitivity can be diagnosed.
    Q_box = coef_bound**2
    Q_diff = holder_L * b ** (gamma - 1.0)
    Q = Q_box + Q_diff

    c = 0.0 if alpha is None or not np.isfinite(alpha) else alpha / (4.0 * Q)

    return Model1D(
        idx=idx,
        b=b,
        b_raw=b_raw,
        b_capped=b_capped,
        R=R,
        coef_bound=coef_bound,
        diff_bound=diff_bound,
        holder_L=holder_L,
        Q_box=Q_box,
        Q_diff=Q_diff,
        Q=Q,
        c=c,
        A=A,
        h=h,
        n=n,
        gamma=gamma,
        alpha=alpha,
    )


# Projected LMC adapted from the supplied diffusion-extension script.

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
    if not np.isfinite(B) or B <= 0 or not np.isfinite(D) or D <= 0:
        raise ValueError("B and D must be finite and positive")
    if max_cycles < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Projection cycles and tolerance must be positive")
    x = np.asarray(y, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Projection input must be a nonempty finite vector")
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
    raise RuntimeError(
        f"Dykstra projection did not converge in {max_cycles} cycles "
        f"(change={change:.3g}, violation={max(box_v, diff_v):.3g}); "
        "increase projection_cycles."
    )


DEFAULT_PRIVACY_BETA = 0.01
DEFAULT_FIXED_RIDGE = 0.001


def ridge_calibration(model, *, ridge=None, ridge_rn=None, ridge_upsilon=None,
                      privacy_beta=None) -> dict:
    """Resolve the fixed or theorem gradient ridge coefficient for this 1D model.

    lambda = min(r_n, 2*R_n**(-2)*log(1 + Upsilon/2)) / |J|,
    with R_n=model.coef_bound, not the localisation radius model.R.
    Default r_n is max(n**(-2*gamma/(1+2*gamma+2)),
                       (n*alpha)**(-2*gamma/(1+gamma+1))).
    With no ridge inputs, use the fixed ridge DEFAULT_FIXED_RIDGE=0.001.
    Explicit theorem inputs select calibration instead; beta=0.01 gives
    Upsilon=beta/(1+exp(alpha)). An explicit ridge selects a fixed value.
    """
    m = model.h.size
    B = float(model.coef_bound)
    if m < 1 or not np.isfinite(B) or B <= 0:
        raise ValueError('Positive coefficient count and box bound required')
    if all(v is None for v in (ridge, ridge_rn, ridge_upsilon, privacy_beta)):
        ridge = DEFAULT_FIXED_RIDGE
    if ridge is not None:
        if any(v is not None for v in (ridge_rn, ridge_upsilon, privacy_beta)):
            raise ValueError('A fixed ridge cannot be combined with theorem ridge inputs')
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError('Fixed ridge must be finite and nonnegative')
        return dict(mode='fixed', ridge=float(ridge), r_n=None, upsilon=None,
                    privacy_beta=None, tv_cap=None, coefficient_radius=B, coefficients=m)
    if ridge_upsilon is None and privacy_beta is None:
        privacy_beta = DEFAULT_PRIVACY_BETA
    if (ridge_upsilon is None) == (privacy_beta is None):
        raise ValueError('Theorem ridge requires exactly one of ridge_upsilon or privacy_beta')
    n, gamma, alpha = model.n, model.gamma, model.alpha
    if n < 1 or not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError('Theorem ridge requires n>=1 and gamma in (0,1]')
    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Theorem ridge requires a finite positive alpha')
    if ridge_rn is None:
        rn = max(math.exp(-2*gamma/(1+2*gamma+2)*math.log(n)),
                 math.exp(-2*gamma/(1+gamma+1)*(math.log(n)+math.log(alpha))))
    else:
        rn = float(ridge_rn)
    if not np.isfinite(rn) or rn <= 0:
        raise ValueError('r_n must be finite and positive')
    if privacy_beta is not None:
        if not np.isfinite(privacy_beta) or not 0 < privacy_beta < 1:
            raise ValueError('privacy_beta must lie in (0,1)')
        # Stable for large positive alpha; do not evaluate exp(alpha).
        ea = math.exp(-alpha)
        upsilon = float(privacy_beta)*ea/(1+ea)
    else:
        upsilon = float(ridge_upsilon)
    if not np.isfinite(upsilon) or upsilon <= 0:
        raise ValueError('Upsilon must be finite and positive (check numerical underflow)')
    cap = 2*math.log1p(upsilon/2)/(B*B)
    value = min(rn, cap)/m
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Calibrated ridge is not representable as a positive finite float')
    return dict(mode='theorem', ridge=value, r_n=rn, upsilon=upsilon,
                privacy_beta=privacy_beta, tv_cap=cap,
                coefficient_radius=B, coefficients=m)


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
    ridge_calibration: dict


def projected_lmc_draw(
    model,
    rng: np.random.Generator,
    n_steps: int,
    ridge: float | None = None,
    step_factor: float = 0.01,
    x_test: np.ndarray | None = None,
    cfg: SimConfig | None = None,
    trace_every: int = 0,
    projection_cycles: int = 80,
    projection_tol: float = 1e-10,
    initial: np.ndarray | None = None,
    trace: np.ndarray | None = None,
    label: str = "",
    report_every: int = 0,
    ridge_rn: float | None = None,
    ridge_upsilon: float | None = None,
    privacy_beta: float | None = None,
) -> tuple[np.ndarray, LMCInfo]:
    """One approximate projected-LMC draw from the ridge-regularised target.

    Potential: c*(theta.T@A@theta/2 + h.T@theta) + ridge*||theta||^2/2.
    Finite step size, finite run length and numerical projection introduce
    approximation error; this routine does not certify TV error or privacy.
    The paper's delta convention is delta=2*eta. The default ridge is fixed
    at 0.001; explicit theorem inputs select calibration from r_n and Upsilon.

    eta = step_factor / L, where L is the smooth-potential gradient Lipschitz
    constant.  The starting point theta=0 is data independent.
    """
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    calibration = ridge_calibration(model, ridge=ridge, ridge_rn=ridge_rn,
                                    ridge_upsilon=ridge_upsilon, privacy_beta=privacy_beta)
    ridge = calibration['ridge']
    if not np.isfinite(step_factor) or not 0 < step_factor < 2:
        raise ValueError("step_factor must lie in (0, 2)")
    if model.c < 0 or not np.isfinite(model.c):
        raise ValueError("model.c must be finite and nonnegative")
    if trace_every < 0 or (trace_every and (x_test is None or cfg is None)):
        raise ValueError("Risk tracing requires x_test, cfg and nonnegative trace_every")
    lam_max = float(np.linalg.eigvalsh(model.A).max())
    Lgrad = model.c * lam_max + ridge
    if not np.isfinite(Lgrad) or Lgrad <= 0:
        raise ValueError("Potential Lipschitz constant must be positive and finite")
    eta = step_factor / Lgrad

    theta = np.zeros(model.h.size) if initial is None else np.asarray(initial, dtype=float).copy()
    if (theta.shape != model.h.shape or not np.isfinite(theta).all()
        or np.max(np.abs(theta)) > model.coef_bound + projection_tol
        or (theta.size > 1 and np.max(np.abs(np.diff(theta))) > model.diff_bound + projection_tol)):
        raise ValueError("Invalid or infeasible initial state")
    if trace is not None and trace.shape != (n_steps, theta.size):
        raise ValueError("Trace must have shape (n_steps, coefficients)")
    if report_every < 0:
        raise ValueError("report_every must be nonnegative")
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
        if trace is not None:
            trace[k - 1] = theta
        if report_every and k % report_every == 0:
            print(f"{label}: {k:,}/{n_steps:,} LMC steps", flush=True)

        if trace_every and (k == 1 or k % trace_every == 0 or k == n_steps):
            trace_steps.append(k)
            if x_test is not None and cfg is not None:
                r = theta_risk(
                    model, theta, x_test, cfg.gamma, cfg.target,
                    cfg.amplitude, cfg.K, cfg.lam,
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
        ridge_calibration=calibration,
    )
    return theta, info



# ============================================================
# Legacy Gibbs kernel (available for comparisons only)
#
# This is NOT the projected-LMC algorithm from the theory section.
# It targets the same constrained exponential-mechanism density pi_D
# in 1D for the simulation scripts that import this module.
#
# The coordinate conditional is:
#   exp(-0.5 a theta_j^2 - b theta_j)
# on an interval determined by box and neighboring-difference
# constraints.  Hence it is a truncated Gaussian (a>0) or a truncated
# exponential/uniform (a=0).
# ============================================================

def _sample_truncated_exponential(
    lo: float,
    hi: float,
    linear_coef: float,
    rng: np.random.Generator,
) -> float:
    """
    Density proportional to exp(-linear_coef*x) on [lo, hi].
    """
    if abs(linear_coef) < 1e-12:
        return float(rng.uniform(lo, hi))

    span = hi - lo

    if linear_coef > 0:
        rate = linear_coef
        den = -np.expm1(-rate * span)
        u = rng.uniform()
        y = -np.log1p(-u * den) / rate
        return float(lo + y)

    rate = -linear_coef
    den = -np.expm1(-rate * span)
    u = rng.uniform()
    y = -np.log1p(-u * den) / rate
    return float(hi - y)


# Historical implementation, kept commented out for reference.
# def _sample_truncated_normal(
#     mean: float,
#     sd: float,
#     lo: float,
#     hi: float,
#     rng: np.random.Generator,
# ) -> float:
#     """
#     Inverse-CDF truncated normal sampler.
#     """
#     a = (lo - mean) / sd
#     b = (hi - mean) / sd

#     pa = ndtr(a)
#     pb = ndtr(b)

#     if pb - pa <= 1e-14:
#         # Numerically extreme tail.  The feasible interval is typically
#         # tiny relative to the Gaussian scale; midpoint is a safe fallback.
#         return float(0.5 * (lo + hi))

#     p = pa + rng.uniform() * (pb - pa)
#     p = min(max(p, 1e-15), 1.0 - 1e-15)
#     x = mean + sd * ndtri(p)
#     return float(min(max(x, lo), hi))

def _sample_truncated_normal(
    mean: float,
    sd: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> float:
    """Sample a truncated normal, including numerically extreme tails."""
    if not np.isfinite(mean) or not np.isfinite(sd) or sd <= 0:
        raise ValueError("Normal mean and scale must be finite; scale must be positive.")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError("Invalid truncation interval.")
    if lo == hi:
        return float(lo)

    a = (lo - mean) / sd
    b = (hi - mean) / sd
    value = truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng)
    if not np.isfinite(value):
        raise FloatingPointError("Truncated-normal sampler returned a non-finite value.")
    return float(np.clip(value, lo, hi))


def gibbs_draws(
    model: Model1D,
    rng: np.random.Generator,
    sweeps: int = 500,
    burn_sweeps: int = 200,
    thin: int = 10,
    n_draws: int = 20,
) -> list[np.ndarray]:
    """
    Coordinate Gibbs draws targeting pi_D(theta).

    The feasible interval for theta_j is the intersection of
      [-B,B],
      [theta_{j-1}-D, theta_{j-1}+D],
      [theta_{j+1}-D, theta_{j+1}+D].
    """
    A = model.A
    h = model.h
    c = model.c
    B = model.coef_bound
    D = model.diff_bound

    m = h.size
    theta = np.zeros(m)
    Atheta = A @ theta
    saved: list[np.ndarray] = []

    for sweep in range(sweeps):
        for j in range(m):
            lo = -B
            hi = B

            if j > 0:
                lo = max(lo, theta[j - 1] - D)
                hi = min(hi, theta[j - 1] + D)

            if j < m - 1:
                lo = max(lo, theta[j + 1] - D)
                hi = min(hi, theta[j + 1] + D)

            # Numerical guard.
            if lo > hi:
                mid = 0.5 * (lo + hi)
                lo = mid
                hi = mid

            old = theta[j]

            # Conditional potential:
            # 0.5*c*A_jj*x^2
            # + c*(h_j + sum_{k != j} A_jk theta_k)*x.
            quad = c * A[j, j]
            lin = c * (
                h[j] + Atheta[j] - A[j, j] * theta[j]
            )

            if hi - lo < 1e-14:
                new = lo
            elif quad > 1e-12:
                mean = -lin / quad
                sd = 1.0 / math.sqrt(quad)
                new = _sample_truncated_normal(
                    mean, sd, lo, hi, rng
                )
            else:
                new = _sample_truncated_exponential(
                    lo, hi, lin, rng
                )

            theta[j] = new
            delta = new - old
            if delta != 0.0:
                Atheta += delta * A[:, j]

        if sweep >= burn_sweeps and (
            (sweep - burn_sweeps) % thin == 0
        ):
            saved.append(theta.copy())

    if not saved:
        saved = [theta.copy()]

    return saved[-n_draws:]


def score_from_theta(
    model: Model1D,
    theta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    psi, _ = spline_design(
        np.asarray(x, dtype=float),
        model.idx,
        model.b,
    )
    return cutoff_chi(x, model.R) * (psi @ theta)


def conditional_mechanism_risk(
    model: Model1D,
    draws: Iterable[np.ndarray],
    x_test: np.ndarray,
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
) -> float:
    """
    Average test risk over MCMC draws from pi_D.

    This estimates E_theta[ ||f_theta-s0||^2_{rho0} | D ].
    We average RISKS over draws, not theta itself.
    """
    s0 = true_score(
        x_test, gamma, target=target, amplitude=amplitude, K=K, lam=lam
    )

    risks = []
    for theta in draws:
        shat = score_from_theta(model, theta, x_test)
        risks.append(np.mean((shat - s0) ** 2))

    return float(np.mean(risks))


def solve_constrained_erm(
    model: Model1D,
    maxiter: int = 2000,
    ftol: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """
    Non-private diagnostic estimator:

        argmin_{theta in Theta_{n,gamma}}
            [ 1/2 theta^T A theta + h^T theta ].

    This is the constrained empirical score-matching ERM, i.e. the
    alpha -> infinity mode corresponding to the same score model.
    It is NOT a second theorem from the paper; it is a diagnostic
    baseline for checking whether the observed finite-sample slope is
    already present without privacy/mechanism randomness.
    """
    A = model.A
    h = model.h
    m = h.size
    B = model.coef_bound
    D = model.diff_bound

    # Scaling by n is irrelevant to the minimizer and improves numerics.
    scale = max(float(np.trace(A)) / max(m, 1), 1.0)
    A0 = A / scale
    h0 = h / scale

    def fun(theta: np.ndarray) -> float:
        return float(0.5 * theta @ A0 @ theta + h0 @ theta)

    def jac(theta: np.ndarray) -> np.ndarray:
        return A0 @ theta + h0

    bounds = Bounds(
        -B * np.ones(m),
        B * np.ones(m),
    )

    constraints = []
    if m > 1:
        # Dmat @ theta = (theta_{j+1} - theta_j)_j.
        Dmat = np.zeros((m - 1, m))
        rows = np.arange(m - 1)
        Dmat[rows, rows] = -1.0
        Dmat[rows, rows + 1] = 1.0
        constraints.append(
            LinearConstraint(
                Dmat,
                -D * np.ones(m - 1),
                D * np.ones(m - 1),
            )
        )

    result = minimize(
        fun,
        x0=np.zeros(m),
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": maxiter,
            "ftol": ftol,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Constrained ERM optimization failed: "
            f"{result.message}"
        )

    theta = np.asarray(result.x, dtype=float)

    info = {
        "erm_success": bool(result.success),
        "erm_nit": int(result.nit),
        "erm_objective": float(result.fun),
    }
    return theta, info


def theta_risk(
    model: Model1D,
    theta: np.ndarray,
    x_test: np.ndarray,
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
) -> float:
    s0 = true_score(
        x_test, gamma, target=target, amplitude=amplitude, K=K, lam=lam
    )
    shat = score_from_theta(model, theta, x_test)
    return float(np.mean((shat - s0) ** 2))




def constraint_stats(
    thetas: Iterable[np.ndarray],
    model: Model1D,
) -> dict:
    """More informative constraint diagnostics than a single max."""
    box_vals = []
    diff_vals = []

    for theta in thetas:
        theta = np.asarray(theta, dtype=float)
        box_vals.extend((np.abs(theta) / model.coef_bound).tolist())
        if theta.size > 1 and model.diff_bound > 0:
            diff_vals.extend(
                (np.abs(np.diff(theta)) / model.diff_bound).tolist()
            )

    box = np.asarray(box_vals, dtype=float)
    diff = np.asarray(diff_vals, dtype=float)

    def summarize_ratio(v: np.ndarray, prefix: str) -> dict:
        if v.size == 0:
            return {
                f"{prefix}_max": np.nan,
                f"{prefix}_median": np.nan,
                f"{prefix}_q95": np.nan,
                f"{prefix}_frac_gt_095": np.nan,
            }
        return {
            f"{prefix}_max": float(np.max(v)),
            f"{prefix}_median": float(np.median(v)),
            f"{prefix}_q95": float(np.quantile(v, 0.95)),
            f"{prefix}_frac_gt_095": float(np.mean(v > 0.95)),
        }

    out = {}
    out.update(summarize_ratio(box, "box"))
    out.update(summarize_ratio(diff, "diff"))
    return out

def fit_loglog_slope(
    n: np.ndarray,
    risk: np.ndarray,
) -> float:
    n = np.asarray(n, dtype=float)
    risk = np.asarray(risk, dtype=float)
    return float(np.polyfit(np.log(n), np.log(risk), 1)[0])


# ============================================================
# One Monte Carlo replicate
# ============================================================

@dataclass
class SimConfig:
    gamma: float = 0.5
    target: str = "multiscale"
    amplitude: float = 0.25
    K: int = 40
    lam: float = 0.25
    bandwidth_factor: float = 0.50
    R_factor: float = 1.25
    coef_bound_factor: float = 1.10
    # Legacy Gibbs settings; not used by one_replicate.
    sweeps: int = 500
    burn_sweeps: int = 200
    thin: int = 10
    n_mcmc_draws: int = 20
    n_test: int = 10000
    # Projected LMC defaults for the current experiments.
    lmc_steps: int = 100000
    # None resolves to DEFAULT_FIXED_RIDGE unless theorem inputs are supplied.
    ridge: float | None = None
    ridge_rn: float | None = None
    ridge_upsilon: float | None = None
    privacy_beta: float | None = None
    lmc_step_factor: float = 0.01
    projection_cycles: int = 80
    projection_tol: float = 1e-10


def one_replicate(
    n: int,
    alpha: float,
    seed: int,
    cfg: SimConfig,
) -> dict:
    rng = np.random.default_rng(seed)

    x = sample_target(
        n=n,
        gamma=cfg.gamma,
        rng=rng,
        target=cfg.target,
        amplitude=cfg.amplitude,
        K=cfg.K,
        lam=cfg.lam,
    )

    model = build_model(
        x=x,
        gamma=cfg.gamma,
        alpha=alpha,
        target=cfg.target,
        amplitude=cfg.amplitude,
        K=cfg.K,
        lam=cfg.lam,
        bandwidth_factor=cfg.bandwidth_factor,
        R_factor=cfg.R_factor,
        coef_bound_factor=cfg.coef_bound_factor,
    )

    theta, lmc_info = projected_lmc_draw(
        model=model, rng=rng, n_steps=cfg.lmc_steps,
        ridge=cfg.ridge, step_factor=cfg.lmc_step_factor,
        ridge_rn=cfg.ridge_rn, ridge_upsilon=cfg.ridge_upsilon,
        privacy_beta=cfg.privacy_beta,
        projection_cycles=cfg.projection_cycles,
        projection_tol=cfg.projection_tol,
    )
    draws = [theta]

    x_test = sample_target(
        n=cfg.n_test,
        gamma=cfg.gamma,
        rng=rng,
        target=cfg.target,
        amplitude=cfg.amplitude,
        K=cfg.K,
        lam=cfg.lam,
    )

    risk = conditional_mechanism_risk(
        model=model,
        draws=draws,
        x_test=x_test,
        gamma=cfg.gamma,
        target=cfg.target,
        amplitude=cfg.amplitude,
        K=cfg.K,
        lam=cfg.lam,
    )

    cstats = constraint_stats(draws, model)
    max_box = max(float(np.max(np.abs(t))) for t in draws)
    max_diff = max(
        float(np.max(np.abs(np.diff(t))))
        if t.size > 1
        else 0.0
        for t in draws
    )

    return {
        "n": n,
        "alpha": alpha,
        "seed": seed,
        "risk": risk,
        "sampler": "projected_lmc",
        "lmc_steps": lmc_info.n_steps,
        "ridge": lmc_info.ridge,
        "ridge_mode": lmc_info.ridge_calibration['mode'],
        "ridge_r_n": lmc_info.ridge_calibration['r_n'],
        "ridge_upsilon": lmc_info.ridge_calibration['upsilon'],
        "lmc_step_size": lmc_info.step_size,
        "lmc_lipschitz": lmc_info.lipschitz,
        "mean_projection_cycles": lmc_info.mean_projection_cycles,
        "max_projection_violation": lmc_info.max_projection_violation,
        "b": model.b,
        "b_raw": model.b_raw,
        "b_capped": model.b_capped,
        "R": model.R,
        "basis_dim": model.idx.size,
        "coef_bound": model.coef_bound,
        "diff_bound": model.diff_bound,
        "Q_box": model.Q_box,
        "Q_diff": model.Q_diff,
        "Q": model.Q,
        "Q_box_fraction": model.Q_box / model.Q,
        "Q_diff_fraction": model.Q_diff / model.Q,
        "max_abs_theta": max_box,
        "max_abs_diff": max_diff,
        "box_ratio": max_box / model.coef_bound,
        "diff_ratio": max_diff / model.diff_bound if model.diff_bound > 0 else np.nan,
        **cstats,
    }


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, as_index=False)["risk"]
    out = g.agg(
        mean_risk="mean",
        sd_risk="std",
        n_rep="count",
    )
    out["se_risk"] = out["sd_risk"] / np.sqrt(out["n_rep"])
    return out
