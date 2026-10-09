#!/usr/bin/env python3
"""
CPU-friendly simulations for:

(1) score-risk vs sample size n;
(2) privacy phase transition vs alpha/alpha_star.

This script implements the 1D version of the paper's spline score model
and exponential-mechanism target.  For the simulation only, it samples
from the SAME constrained exponential-mechanism target with a
coordinate Gibbs sampler rather than projected LMC.  This is much
cheaper on CPU and isolates the statistical rate / privacy regime.

Dependencies:
    numpy
    scipy
    pandas
    matplotlib

Example:
    python private_score_experiments.py --experiment both --quick
    python private_score_experiments.py --experiment rate --full
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import ndtr, ndtri
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
    K: int = 10,
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
    K: int = 10,
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


def build_model(
    x: np.ndarray,
    gamma: float,
    alpha: float | None,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 10,
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
    )


# ============================================================
# CPU-friendly exact-target Gibbs kernel
#
# This is NOT the projected-LMC algorithm from the theory section.
# It targets the same constrained exponential-mechanism density pi_D
# in 1D.  It is used here only to make Experiments 1-2 feasible on CPU.
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


def _sample_truncated_normal(
    mean: float,
    sd: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> float:
    """
    Inverse-CDF truncated normal sampler.
    """
    a = (lo - mean) / sd
    b = (hi - mean) / sd

    pa = ndtr(a)
    pb = ndtr(b)

    if pb - pa <= 1e-14:
        # Numerically extreme tail.  The feasible interval is typically
        # tiny relative to the Gaussian scale; midpoint is a safe fallback.
        return float(0.5 * (lo + hi))

    p = pa + rng.uniform() * (pb - pa)
    p = min(max(p, 1e-15), 1.0 - 1e-15)
    x = mean + sd * ndtri(p)
    return float(min(max(x, lo), hi))


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
    K: int = 10
    lam: float = 0.25
    bandwidth_factor: float = 0.50
    R_factor: float = 1.25
    coef_bound_factor: float = 1.10
    sweeps: int = 500
    burn_sweeps: int = 200
    thin: int = 10
    n_mcmc_draws: int = 20
    n_test: int = 10000


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

    draws = gibbs_draws(
        model=model,
        rng=rng,
        sweeps=cfg.sweeps,
        burn_sweeps=cfg.burn_sweeps,
        thin=cfg.thin,
        n_draws=cfg.n_mcmc_draws,
    )

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


# ============================================================
# Experiment 1:
# score risk vs n
# ============================================================

def run_rate_experiment(
    outdir: Path,
    cfg: SimConfig,
    n_values: list[int],
    alpha_values: list[float],
    reps: int,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Experiment 1 with an added non-private constrained ERM diagnostic.

    For every (n, replicate), the SAME training sample and SAME test
    sample are reused for all alpha values and for the ERM.  This makes
    the comparison paired and substantially easier to interpret.

    Outputs:
      rate_raw.csv
      rate_summary.csv
      rate_diagnostics.csv
      rate_empirical_slopes.csv
      experiment1_rate_vs_n.png/pdf
    """
    outdir.mkdir(parents=True, exist_ok=True)

    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(len(n_values) * reps)

    rows = []
    k = 0

    for n in n_values:
        for rep in range(reps):
            seed = int(child_seeds[k].generate_state(1)[0])
            k += 1
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
            x_test = sample_target(
                n=cfg.n_test,
                gamma=cfg.gamma,
                rng=rng,
                target=cfg.target,
                amplitude=cfg.amplitude,
                K=cfg.K,
                lam=cfg.lam,
            )

            # --------------------------------------------------------
            # Non-private constrained score-matching ERM.
            # Its bandwidth uses only the non-private term.
            # --------------------------------------------------------
            erm_model = build_model(
                x=x,
                gamma=cfg.gamma,
                alpha=None,
                target=cfg.target,
                amplitude=cfg.amplitude,
                K=cfg.K,
                lam=cfg.lam,
                bandwidth_factor=cfg.bandwidth_factor,
                R_factor=cfg.R_factor,
                coef_bound_factor=cfg.coef_bound_factor,
            )

            theta_erm, erm_info = solve_constrained_erm(erm_model)
            erm_risk = theta_risk(
                model=erm_model,
                theta=theta_erm,
                x_test=x_test,
                gamma=cfg.gamma,
                target=cfg.target,
                amplitude=cfg.amplitude,
                K=cfg.K,
                lam=cfg.lam,
            )

            erm_cstats = constraint_stats([theta_erm], erm_model)
            erm_max_box = float(np.max(np.abs(theta_erm)))
            erm_max_diff = (
                float(np.max(np.abs(np.diff(theta_erm))))
                if theta_erm.size > 1 else 0.0
            )

            rows.append({
                "estimator": "ERM",
                "alpha": np.nan,
                "n": n,
                "rep": rep,
                "seed": seed,
                "risk": erm_risk,
                "b": erm_model.b,
                "b_raw": erm_model.b_raw,
                "b_capped": erm_model.b_capped,
                "R": erm_model.R,
                "basis_dim": erm_model.idx.size,
                "coef_bound": erm_model.coef_bound,
                "diff_bound": erm_model.diff_bound,
                "Q_box": erm_model.Q_box,
                "Q_diff": erm_model.Q_diff,
                "Q": erm_model.Q,
                "Q_box_fraction": erm_model.Q_box / erm_model.Q,
                "Q_diff_fraction": erm_model.Q_diff / erm_model.Q,
                "max_abs_theta": erm_max_box,
                "max_abs_diff": erm_max_diff,
                "box_ratio": erm_max_box / erm_model.coef_bound,
                "diff_ratio": (
                    erm_max_diff / erm_model.diff_bound
                    if erm_model.diff_bound > 0 else np.nan
                ),
                **erm_cstats,
                **erm_info,
            })

            print(
                f"[rate] n={n:6d} rep={rep+1:2d}/{reps} "
                f"ERM risk={erm_risk:.5g}"
            )

            # --------------------------------------------------------
            # Private exponential-mechanism estimator.
            # --------------------------------------------------------
            for alpha in alpha_values:
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

                draws = gibbs_draws(
                    model=model,
                    rng=rng,
                    sweeps=cfg.sweeps,
                    burn_sweeps=cfg.burn_sweeps,
                    thin=cfg.thin,
                    n_draws=cfg.n_mcmc_draws,
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
                max_box = max(
                    float(np.max(np.abs(t)))
                    for t in draws
                )
                max_diff = max(
                    float(np.max(np.abs(np.diff(t))))
                    if t.size > 1 else 0.0
                    for t in draws
                )

                rows.append({
                    "estimator": f"alpha={alpha:g}",
                    "alpha": alpha,
                    "n": n,
                    "rep": rep,
                    "seed": seed,
                    "risk": risk,
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
                    "diff_ratio": (
                        max_diff / model.diff_bound
                        if model.diff_bound > 0 else np.nan
                    ),
                    "erm_success": np.nan,
                    "erm_nit": np.nan,
                    "erm_objective": np.nan,
                    **cstats,
                })

                print(
                    f"       alpha={alpha:5.2f} "
                    f"risk={risk:.5g}  "
                    f"b={model.b:.4f}  "
                    f"|J|={model.idx.size:3d}  "
                    f"Q={model.Q:.3f}  "
                    f"box={max_box/model.coef_bound:.2f}  "
                    f"diff={max_diff/model.diff_bound:.2f}"
                )

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(["estimator", "n"], as_index=False)["risk"]
        .agg(mean_risk="mean", sd_risk="std", n_rep="count")
    )
    summary["se_risk"] = (
        summary["sd_risk"] / np.sqrt(summary["n_rep"])
    )

    diagnostics = (
        raw.groupby(["estimator", "n"], as_index=False)
        .agg(
            mean_b=("b", "mean"),
            mean_b_raw=("b_raw", "mean"),
            cap_rate=("b_capped", "mean"),
            mean_R=("R", "mean"),
            mean_basis_dim=("basis_dim", "mean"),
            mean_Q_box=("Q_box", "mean"),
            mean_Q_diff=("Q_diff", "mean"),
            mean_Q=("Q", "mean"),
            mean_Q_box_fraction=("Q_box_fraction", "mean"),
            mean_Q_diff_fraction=("Q_diff_fraction", "mean"),
            mean_box_ratio=("box_ratio", "mean"),
            mean_diff_ratio=("diff_ratio", "mean"),
            mean_box_median=("box_median", "mean"),
            mean_box_q95=("box_q95", "mean"),
            mean_box_frac_gt_095=("box_frac_gt_095", "mean"),
            mean_diff_median=("diff_median", "mean"),
            mean_diff_q95=("diff_q95", "mean"),
            mean_diff_frac_gt_095=("diff_frac_gt_095", "mean"),
        )
    )

    raw.to_csv(outdir / "rate_raw.csv", index=False)
    summary.to_csv(outdir / "rate_summary.csv", index=False)
    diagnostics.to_csv(outdir / "rate_diagnostics.csv", index=False)

    # ------------------------------------------------------------
    # Empirical slopes: all n's and largest three n's.
    # ------------------------------------------------------------
    slope_rows = []

    ordered_estimators = ["ERM"] + [
        f"alpha={a:g}" for a in alpha_values
    ]

    for estimator in ordered_estimators:
        s = summary[
            summary["estimator"] == estimator
        ].sort_values("n")

        n_arr = s["n"].to_numpy(dtype=float)
        risk_arr = s["mean_risk"].to_numpy(dtype=float)

        if len(s) >= 2:
            all_slope = fit_loglog_slope(n_arr, risk_arr)
        else:
            all_slope = np.nan

        tail = s.tail(min(3, len(s)))
        if len(tail) >= 2:
            tail_slope = fit_loglog_slope(
                tail["n"].to_numpy(dtype=float),
                tail["mean_risk"].to_numpy(dtype=float),
            )
        else:
            tail_slope = np.nan

        slope_rows.append({
            "estimator": estimator,
            "slope_all_n": all_slope,
            "slope_largest_3_n": tail_slope,
        })

    slopes = pd.DataFrame(slope_rows)

    gamma = cfg.gamma
    d = 1
    slopes["theory_nonprivate_slope"] = (
        -2.0 * gamma / (d + 2.0 * gamma + 2.0)
    )
    slopes["theory_privacy_dominated_slope"] = (
        -2.0 * gamma / (d + gamma + 1.0)
    )

    slopes.to_csv(
        outdir / "rate_empirical_slopes.csv",
        index=False,
    )

    print("\nEmpirical log-log slopes")
    print(slopes.to_string(index=False))
    print()

    # ------------------------------------------------------------
    # Plot.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.5))

    for estimator in ordered_estimators:
        s = summary[
            summary["estimator"] == estimator
        ].sort_values("n")

        if estimator == "ERM":
            label = "non-private constrained ERM"
            marker = "s"
        else:
            alpha = float(estimator.split("=")[1])
            label = fr"$\alpha={alpha:g}$"
            marker = "o"

        ax.errorbar(
            s["n"],
            s["mean_risk"],
            yerr=s["se_risk"],
            marker=marker,
            capsize=3,
            label=label,
        )

    # One non-private reference slope, anchored at the largest-n ERM.
    erm_s = summary[
        summary["estimator"] == "ERM"
    ].sort_values("n")

    if not erm_s.empty:
        n_grid = np.asarray(n_values, dtype=float)
        np_exp = 2.0 * gamma / (
            d + 2.0 * gamma + 2.0
        )
        n_anchor = float(erm_s["n"].iloc[-1])
        y_anchor = float(erm_s["mean_risk"].iloc[-1])

        ref = y_anchor * (n_grid / n_anchor) ** (-np_exp)
        ax.plot(
            n_grid,
            ref,
            linestyle="--",
            linewidth=1.8,
            label=fr"reference $n^{{-{np_exp:.3g}}}$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel(
        r"score risk  $\mathbb{E}\|\hat s-s_0\|_{\rho_0}^2$"
    )
    ax.set_title(
        fr"Experiment 1 + diagnostic "
        fr"($\gamma={cfg.gamma}$)"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        outdir / "experiment1_rate_vs_n.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "experiment1_rate_vs_n.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Sensitivity decomposition diagnostic.  We show the fraction of
    # Q coming from the coefficient/box term.  The derivative term is
    # exactly one minus this fraction.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for estimator in ordered_estimators:
        dsub = diagnostics[
            diagnostics["estimator"] == estimator
        ].sort_values("n")
        ax.plot(
            dsub["n"],
            dsub["mean_Q_box_fraction"],
            marker="o",
            label=estimator,
        )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("sample size n")
    ax.set_ylabel(r"coefficient-bound share  $Q_{box}/Q$")
    ax.set_title("Finite-sample sensitivity decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        outdir / "experiment1_Q_box_fraction.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "experiment1_Q_box_fraction.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return raw, summary


# ============================================================
# Experiment 2:
# finite-sample privacy regimes
#
# We still use the asymptotic normalization
#
#   alpha_star = n^{-(gamma+1)/(d+2gamma+2)},
#
# but DO NOT draw an exact transition at alpha/alpha_star=1.
# The theorem is only up to polylogarithmic factors, and the actual
# finite-sample exponential mechanism depends on Q_{n,gamma}.
# ============================================================

def run_phase_transition_experiment(
    outdir: Path,
    cfg: SimConfig,
    n_values: list[int],
    alpha_ratios: list[float],
    reps: int,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)

    gamma = cfg.gamma
    d = 1

    ss = np.random.SeedSequence(master_seed)
    child_seeds = ss.spawn(len(n_values) * reps)

    rows = []
    k = 0

    for n in n_values:
        alpha_star = n ** (
            -(gamma + 1.0) / (d + 2.0 * gamma + 2.0)
        )

        for rep in range(reps):
            seed = int(child_seeds[k].generate_state(1)[0])
            k += 1
            rng = np.random.default_rng(seed)

            # Paired comparison: all privacy levels use the same D and
            # the same test sample within a replicate.
            x = sample_target(
                n=n,
                gamma=cfg.gamma,
                rng=rng,
                target=cfg.target,
                amplitude=cfg.amplitude,
                K=cfg.K,
                lam=cfg.lam,
            )
            x_test = sample_target(
                n=cfg.n_test,
                gamma=cfg.gamma,
                rng=rng,
                target=cfg.target,
                amplitude=cfg.amplitude,
                K=cfg.K,
                lam=cfg.lam,
            )

            for ratio in alpha_ratios:
                alpha = ratio * alpha_star

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

                draws = gibbs_draws(
                    model=model,
                    rng=rng,
                    sweeps=cfg.sweeps,
                    burn_sweeps=cfg.burn_sweeps,
                    thin=cfg.thin,
                    n_draws=cfg.n_mcmc_draws,
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

                rows.append({
                    "n": n,
                    "rep": rep,
                    "seed": seed,
                    "alpha": alpha,
                    "alpha_star": alpha_star,
                    "alpha_ratio": ratio,
                    "risk": risk,
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
                    **cstats,
                })

                cap_flag = " CAP" if model.b_capped else ""
                print(
                    f"[phase] n={n:6d} rep={rep+1:2d}/{reps} "
                    f"a/a*={ratio:6.3f} risk={risk:.5g} "
                    f"b={model.b:.4f} raw={model.b_raw:.4f}{cap_flag} "
                    f"Qbox/Q={model.Q_box/model.Q:.3f}"
                )

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(
            ["n", "alpha_ratio", "alpha", "alpha_star"],
            as_index=False,
        )
        .agg(
            mean_risk=("risk", "mean"),
            sd_risk=("risk", "std"),
            n_rep=("risk", "count"),
            mean_b=("b", "mean"),
            mean_b_raw=("b_raw", "mean"),
            cap_rate=("b_capped", "mean"),
            mean_Q=("Q", "mean"),
            mean_Q_box_fraction=("Q_box_fraction", "mean"),
            mean_Q_diff_fraction=("Q_diff_fraction", "mean"),
            mean_diff_median=("diff_median", "mean"),
            mean_diff_q95=("diff_q95", "mean"),
            mean_diff_frac_gt_095=("diff_frac_gt_095", "mean"),
        )
    )
    summary["se_risk"] = summary["sd_risk"] / np.sqrt(summary["n_rep"])

    normalized_parts = []
    for n in n_values:
        s_n = summary[summary["n"] == n].copy()
        plateau = s_n.loc[s_n["alpha_ratio"].idxmax(), "mean_risk"]
        s_n["normalized_risk"] = s_n["mean_risk"] / plateau
        s_n["normalized_se"] = s_n["se_risk"] / plateau
        normalized_parts.append(s_n)

    normalized = pd.concat(normalized_parts, ignore_index=True)

    raw.to_csv(outdir / "phase_raw.csv", index=False)
    normalized.to_csv(outdir / "phase_summary.csv", index=False)

    # ------------------------------------------------------------
    # Main regime plot.  No exact theory curve / exact crossover line.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.3, 5.4))

    any_capped = False
    any_capped = False
    for n in n_values:
        s_n = normalized[
            normalized["n"] == n
        ].sort_values("alpha_ratio")

        ax.errorbar(
            s_n["alpha_ratio"],
            s_n["normalized_risk"],
            yerr=s_n["normalized_se"],
            marker="o",
            capsize=3,
            label=fr"$n={n}$",
        )

        capped = s_n[s_n["cap_rate"] > 0]
        if not capped.empty:
            any_capped = True
            ax.scatter(
                capped["alpha_ratio"],
                capped["normalized_risk"],
                marker="x",
                s=80,
            )

    if any_capped:
        ax.scatter([], [], marker="x", s=80, label="bandwidth cap active")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"normalized privacy level  $\alpha/\alpha_\star"
    )
    ax.set_ylabel("risk / largest-alpha risk")
    ax.set_title(
        fr"Experiment 2: finite-sample privacy regimes "
        fr"($\gamma={cfg.gamma}$)"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        outdir / "experiment2_phase_transition.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "experiment2_phase_transition.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ------------------------------------------------------------
    # Sensitivity diagnostic.  Q_diff/Q = 1 - Q_box/Q.
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.3, 5.2))

    for n in n_values:
        s_n = normalized[
            normalized["n"] == n
        ].sort_values("alpha_ratio")
        ax.plot(
            s_n["alpha_ratio"],
            s_n["mean_Q_box_fraction"],
            marker="o",
            label=fr"$n={n}$",
        )

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"normalized privacy level  $\alpha/\alpha_\star$")
    ax.set_ylabel(r"coefficient-bound share  $Q_{box}/Q$")
    ax.set_title("Why finite-sample privacy can differ from asymptotics")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        outdir / "experiment2_Q_box_fraction.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "experiment2_Q_box_fraction.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return raw, normalized


# ============================================================
# CLI
# ============================================================

def make_config(quick: bool, full: bool) -> SimConfig:
    if full:
        return SimConfig(
            sweeps=800,
            burn_sweeps=300,
            thin=20,
            n_mcmc_draws=20,
            n_test=20000,
        )

    # quick/default
    return SimConfig(
        sweeps=300,
        burn_sweeps=120,
        thin=10,
        n_mcmc_draws=10,
        n_test=5000,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["rate", "phase", "both"],
        default="both",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small CPU-friendly run",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="larger paper-style run",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--target",
        choices=["multiscale", "cusp"],
        default="multiscale",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.25,
        help="amplitude a for the multiscale target",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=10,
        help="highest dyadic frequency level for the multiscale target",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--rate-reps",
        type=int,
        default=None,
        help="override the number of Monte Carlo replicates for Experiment 1",
    )
    parser.add_argument(
        "--phase-reps",
        type=int,
        default=None,
        help="override the number of Monte Carlo replicates for Experiment 2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261008,
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="sim_results_v3",
    )

    args = parser.parse_args()

    cfg = make_config(
        quick=args.quick,
        full=args.full,
    )
    cfg.gamma = args.gamma
    cfg.target = args.target
    cfg.amplitude = args.amplitude
    cfg.K = args.K
    cfg.lam = args.lam

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [asdict(cfg)]
    ).to_csv(
        outdir / "simulation_config.csv",
        index=False,
    )

    if args.full:
        rate_n = [500, 1000, 2000, 4000, 8000, 16000]
        rate_alpha = [0.25, 0.5, 1.0, 2.0]
        rate_reps = 20

        phase_n = [2000, 4000, 8000, 16000, 32000]
        phase_ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        phase_reps = 10
    else:
        rate_n = [400, 800, 1600, 3200, 6400, 12800]
        rate_alpha = [0.25, 0.5, 1.0, 2.0]
        rate_reps = 5

        phase_n = [2000, 8000, 16000]
        phase_ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        phase_reps = 3

    if args.rate_reps is not None:
        rate_reps = args.rate_reps
    if args.phase_reps is not None:
        phase_reps = args.phase_reps

    if args.experiment in ("rate", "both"):
        run_rate_experiment(
            outdir=outdir,
            cfg=cfg,
            n_values=rate_n,
            alpha_values=rate_alpha,
            reps=rate_reps,
            master_seed=args.seed,
        )

    if args.experiment in ("phase", "both"):
        run_phase_transition_experiment(
            outdir=outdir,
            cfg=cfg,
            n_values=phase_n,
            alpha_ratios=phase_ratios,
            reps=phase_reps,
            master_seed=args.seed + 1,
        )


if __name__ == "__main__":
    main()
