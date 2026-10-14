"""Core 1D implementation for the DP score-estimation simulation suite.

The statistical experiments use a coordinate Gibbs sampler targeting the
same constrained exponential-mechanism density as in the paper.  This is
used as a CPU-friendly reference mechanism; projected Langevin is treated
separately in the sampler-diagnostics experiment.
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


def mixture_score(
    x: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Score of 0.5 N(-mu,1)+0.5 N(mu,1)."""
    x = np.asarray(x, dtype=float)
    return -x + mu * np.tanh(mu * x)


def true_score(
    x: np.ndarray,
    gamma: float,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 10,
    lam: float = 0.25,
    mixture_mu: float = 2.0,
) -> np.ndarray:
    if target == "multiscale":
        return multiscale_score(x, gamma, amplitude, K)
    if target == "cusp":
        return cusp_score(x, gamma, lam)
    if target == "mixture":
        return mixture_score(x, mixture_mu)
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
    mixture_mu: float = 2.0,
) -> float:
    if target == "multiscale":
        geom = sum(2.0 ** (-k * gamma) for k in range(K + 1))
        return xmax + amplitude * geom
    if target == "cusp":
        return xmax + lam * (1.0 + gamma) * xmax**gamma
    if target == "mixture":
        return xmax + abs(mixture_mu)
    raise ValueError(f"unknown target: {target}")


def holder_constant(
    gamma: float,
    target: str,
    amplitude: float,
    K: int,
    lam: float,
    mixture_mu: float = 2.0,
) -> float:
    if target == "multiscale":
        return multiscale_holder_constant(gamma, amplitude, K)
    if target == "cusp":
        return 1.0 + 2.0 * lam * (1.0 + gamma)
    if target == "mixture":
        # The mixture score is Lipschitz.  On |x-y|<=1, a Lipschitz
        # bound is also a gamma-Holder bound for every gamma in (0,1].
        return 1.0 + mixture_mu**2
    raise ValueError(f"unknown target: {target}")


def sample_target(
    n: int,
    gamma: float,
    rng: np.random.Generator,
    target: str = "multiscale",
    amplitude: float = 0.25,
    K: int = 10,
    lam: float = 0.25,
    mixture_mu: float = 2.0,
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

    if target == "mixture":
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        return rng.normal(loc=signs * mixture_mu, scale=1.0, size=n)

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
    mixture_mu: float = 2.0,
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
        mixture_mu=mixture_mu,
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
        mixture_mu=mixture_mu,
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
    mixture_mu: float = 2.0,
) -> float:
    """
    Average test risk over MCMC draws from pi_D.

    This estimates E_theta[ ||f_theta-s0||^2_{rho0} | D ].
    We average RISKS over draws, not theta itself.
    """
    s0 = true_score(
        x_test, gamma, target=target, amplitude=amplitude, K=K, lam=lam, mixture_mu=mixture_mu
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
    mixture_mu: float = 2.0,
) -> float:
    s0 = true_score(
        x_test, gamma, target=target, amplitude=amplitude, K=K, lam=lam, mixture_mu=mixture_mu
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
    mixture_mu: float = 2.0
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
        mixture_mu=cfg.mixture_mu,
    )

    model = build_model(
        x=x,
        gamma=cfg.gamma,
        alpha=alpha,
        target=cfg.target,
        amplitude=cfg.amplitude,
        K=cfg.K,
        lam=cfg.lam,
        mixture_mu=cfg.mixture_mu,
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
        mixture_mu=cfg.mixture_mu,
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
        mixture_mu=cfg.mixture_mu,
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


